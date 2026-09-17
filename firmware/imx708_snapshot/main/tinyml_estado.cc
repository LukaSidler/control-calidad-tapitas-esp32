/* Implementacion de tinyml_estado.h. Este archivo es .cc (no .c) porque
 * TFLite Micro es una API C++; tinyml_estado.h expone un wrapper extern "C"
 * para que imx708_snapshot_main.c (que es C puro) lo pueda llamar sin
 * enterarse de nada de esto.
 *
 * Preprocesamiento: tiene que reproducir EXACTAMENTE lo que hace
 * process_sample() en dataset_tapitas/modelo/train.py (sin augmentation,
 * ver la rama augment=False), para que las probabilidades de aca coincidan
 * con las que salieron en el test set offline (ver threshold_results.txt):
 *   1. Recorte fijo 1120x1120 centrado en el medio del frame.
 *   2. Resize a 128x128 -- en Python es cv2.resize(..., INTER_AREA), que
 *      promedia area por bloques; aca se aproxima con un promedio de
 *      bloques equivalente (ver downsample_block_avg_gray).
 *   3. Escala de grises. OJO: el orden entrenamiento es resize-color
 *      PRIMERO, gris DESPUES; aca se hace gris primero y despues se
 *      promedia -- son matematicamente equivalentes porque el gris es una
 *      combinacion lineal de R/G/B y el promedio conmuta con eso (el
 *      promedio de la combinacion lineal es igual a la combinacion lineal
 *      de los promedios). No es una aproximacion, da el mismo numero.
 *   4. Cuantizar a int8 con la escala/zero-point que trae el propio
 *      modelo (NUNCA hardcodeados -- el modelo entrenado no aplica /255 a
 *      mano: tiene una capa Rescaling(1/255) adentro, y esa escala queda
 *      absorbida en la cuantizacion del tensor de entrada al convertir a
 *      TFLite. Por eso aca se cuantiza el valor de gris CRUDO (0-255), no
 *      dividido por 255 -- si se hiciera dos veces la salida del modelo
 *      queda mal).
 *
 * IMPORTANTE -- requiere kernels ANSI-C de esp-nn (sdkconfig):
 *   Los kernels "optimizados" (PIE/SIMD) de esp-nn para ESP32-P4 dan
 *   resultados numericamente INCORRECTOS para este modelo (probado:
 *   compararando el mismo input cuantizado contra una replica en Python/
 *   tflite_runtime, el ESP32 con kernels optimizados daba clasificaciones
 *   opuestas). El sdkconfig de este proyecto tiene que tener:
 *     CONFIG_NN_ANSI_C=y
 *     # CONFIG_NN_OPTIMIZED is not set
 *     CONFIG_NN_OPTIMIZATIONS=0
 *   Esto se configura en "ESP-NN" dentro de idf.py menuconfig. Si algun dia
 *   un `idf.py menuconfig` o una actualizacion de esp-nn lo resetea a
 *   CONFIG_NN_OPTIMIZED=y, el sintoma va a ser el mismo de antes: prob_rota
 *   pegado cerca de un extremo (0.9+) sin importar la tapita real. Ver
 *   NOTAS_TINYML.md en la raiz del proyecto (imx708_snapshot/) para el
 *   detalle completo de como se diagnostico esto.
 */
#include "tinyml_estado.h"
#include "tapita_model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include <math.h>
#include <string.h>
#include <inttypes.h>

static const char *TAG = "tinyml";

namespace {

constexpr int kCropSize = 1120;
constexpr int kImgSize = 128;

/* Punto de partida generoso: el feature map mas grande de la red (salida
 * de la primera Conv2D, 128x128x16 en int8) ya son 256KB, y TFLM necesita
 * tener varios tensores vivos a la vez ademas de eso. Se arranca grande a
 * proposito y se loguea el uso real con interpreter->arena_used_bytes()
 * despues de AllocateTensors() -- una vez que ese numero salga en el log,
 * se puede achicar esta constante al valor real (+ margen) si se quiere
 * recuperar PSRAM para otra cosa. No hay apuro: en el P4 con PSRAM esto es
 * un porcentaje chico de la memoria disponible. */
constexpr int kTensorArenaSize = 512 * 1024;

const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;
uint8_t *g_tensor_arena = nullptr;
bool g_ready = false;

/* Desempaqueta un pixel RGB565 (mismo layout que sample_average_rgb_region
 * en imx708_snapshot_main.c) a gris 0-255 usando los mismos pesos que
 * cv2.cvtColor(..., COLOR_BGR2GRAY) (BT.601: 0.299 R + 0.587 G + 0.114 B). */
static inline float rgb565_to_gray(uint16_t p)
{
    uint8_t r5 = (p >> 11) & 0x1F;
    uint8_t g6 = (p >> 5) & 0x3F;
    uint8_t b5 = p & 0x1F;
    float r = (r5 * 255) / 31.0f;
    float g = (g6 * 255) / 63.0f;
    float b = (b5 * 255) / 31.0f;
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

/* Recorta kCropSize x kCropSize centrado en el frame (con clamp tipo
 * "replicate border" para los bordes que se salen, igual que
 * cv2.copyMakeBorder(..., BORDER_REPLICATE) en fixed_crop() de train.py --
 * pasa con el alto: el frame mide 1080 y el recorte 1120, sobran ~40px
 * verticales que se resuelven repitiendo la fila de borde), reduce a
 * kImgSize x kImgSize promediando por bloques (equivalente a
 * cv2.INTER_AREA), y cuantiza directo a int8 en el tensor de entrada. */
static void preprocess_into_input(const uint8_t *frame, uint32_t frame_w, uint32_t frame_h,
                                   int8_t *out_int8, float in_scale, int32_t in_zp)
{
    const int crop_x0 = ((int)frame_w - kCropSize) / 2;
    const int crop_y0 = ((int)frame_h - kCropSize) / 2;

    for (int oy = 0; oy < kImgSize; oy++) {
        /* Limites del bloque fuente para esta fila de salida, en
         * coordenadas del RECORTE (0..kCropSize). float a proposito: la
         * relacion 1120/128 = 8.75 no es entera. */
        int sy0 = (int)floorf(oy * (float)kCropSize / kImgSize);
        int sy1 = (int)floorf((oy + 1) * (float)kCropSize / kImgSize);
        if (sy1 <= sy0) sy1 = sy0 + 1;

        for (int ox = 0; ox < kImgSize; ox++) {
            int sx0 = (int)floorf(ox * (float)kCropSize / kImgSize);
            int sx1 = (int)floorf((ox + 1) * (float)kCropSize / kImgSize);
            if (sx1 <= sx0) sx1 = sx0 + 1;

            float sum = 0.0f;
            int count = 0;
            for (int cy = sy0; cy < sy1; cy++) {
                /* Coordenada real en el frame, con clamp (replicate). */
                int fy = crop_y0 + cy;
                if (fy < 0) fy = 0;
                if (fy >= (int)frame_h) fy = (int)frame_h - 1;
                const uint16_t *row = (const uint16_t *)(frame + (size_t)fy * frame_w * 2);

                for (int cx = sx0; cx < sx1; cx++) {
                    int fx = crop_x0 + cx;
                    if (fx < 0) fx = 0;
                    if (fx >= (int)frame_w) fx = (int)frame_w - 1;
                    sum += rgb565_to_gray(row[fx]);
                    count++;
                }
            }
            float gray = sum / (float)count;

            /* Cuantizacion: mismo criterio que eval_tflite.py /
             * threshold_analysis.py -- x_q = round(gray/scale) + zero_point,
             * gray SIN dividir por 255 (ver comentario arriba del archivo). */
            int32_t q = (int32_t)lroundf(gray / in_scale) + in_zp;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            out_int8[oy * kImgSize + ox] = (int8_t)q;
        }
    }
}

} // namespace

void tinyml_init(void)
{
    g_model = tflite::GetModel(g_tapita_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "version de schema del modelo (%" PRIu32 ") no coincide con la soportada (%d)",
                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    g_tensor_arena = (uint8_t *)heap_caps_malloc_prefer(
        kTensorArenaSize, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, MALLOC_CAP_8BIT);
    if (g_tensor_arena == nullptr) {
        ESP_LOGE(TAG, "no se pudo reservar el tensor arena (%d bytes)", kTensorArenaSize);
        return;
    }

    /* Los 5 ops que realmente usa tapita_model_int8.tflite (confirmado
     * inspeccionando el flatbuffer del modelo, no una lista generica):
     * CONV_2D, MAX_POOL_2D, REDUCE_MAX (asi es como el conversor de TFLite
     * traduce GlobalMaxPooling2D), FULLY_CONNECTED, LOGISTIC (la sigmoid
     * final). Si el dia de mañana se reentrena con otra arquitectura y
     * cambian las capas, esta lista hay que revisarla -- un op faltante
     * tira error de AllocateTensors(), no un crash silencioso. */
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReduceMax();
    resolver.AddFullyConnected();
    resolver.AddLogistic();

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    TfLiteStatus status = g_interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() fallo (arena de %d bytes insuficiente?)", kTensorArenaSize);
        return;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);

    if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "el modelo no es int8 (in=%d out=%d) -- se esperaba full-integer",
                 g_input->type, g_output->type);
        return;
    }

    g_ready = true;
    ESP_LOGI(TAG, "listo. arena usada: %d / %d bytes. input scale=%.6f zp=%" PRId32
             "  output scale=%.6f zp=%" PRId32,
             (int)g_interpreter->arena_used_bytes(), kTensorArenaSize,
             g_input->params.scale, (int32_t)g_input->params.zero_point,
             g_output->params.scale, (int32_t)g_output->params.zero_point);
}

bool tinyml_classify_estado(const uint8_t *frame_rgb565, uint32_t frame_w,
                             uint32_t frame_h, float *prob_rota_out)
{
    if (!g_ready) {
        return false;
    }

    int64_t t0 = esp_timer_get_time();

    preprocess_into_input(frame_rgb565, frame_w, frame_h, g_input->data.int8,
                           g_input->params.scale, (int32_t)g_input->params.zero_point);

    int64_t t1 = esp_timer_get_time();

    if (g_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() fallo");
        return false;
    }

    int64_t t2 = esp_timer_get_time();

    int8_t raw = g_output->data.int8[0];
    float prob = (raw - g_output->params.zero_point) * g_output->params.scale;
    *prob_rota_out = prob;

    ESP_LOGI(TAG, "  [tinyml] prob_rota=%.4f (preproc=%lldms invoke=%lldms)",
             prob, (long long)((t1 - t0) / 1000), (long long)((t2 - t1) / 1000));

    return true;
}
