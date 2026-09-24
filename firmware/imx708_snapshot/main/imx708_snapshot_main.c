/*
 * Captura a demanda + clasificacion de color por HSV, sobre el framework
 * IMX708 (esp_cam_sensor_imx). Basado en imx708_snapshot, con la logica de
 * multi-resolucion/sweep quitada y reemplazada por disparo manual (c/i).
 *
 * Agregado: control manual de balance de blancos via el nodo ISP
 * (/dev/video20, V4L2_CID_USER_ESP_ISP_WB), para eliminar el cast de
 * color que el AWB automatico metia segun el fondo de la escena.
 * Flujo: 'm' apaga las estadisticas de AWB y activa ganancia manual,
 * 'r'/'R' y 'b'/'B' ajustan ganancia roja/azul en caliente, 'c'/'i'
 * capturan y muestran el RGB promedio resultante para ir afinando a ojo.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "esp_log.h"
#include "esp_check.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "linux/videodev2.h"
#include "esp_video_init.h"
#include "esp_video_device.h"
#include "esp_video_ioctl.h"
#include "esp_video_isp_ioctl.h"
#include "esp_video_isp_pipeline.h"
#include "esp_random.h"
#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "imx708.h"
#include "esp_cache.h"
#include "esp_heap_caps.h"
#include "driver/jpeg_encode.h"
#include "imx_serial_img.h"
#include "tinyml_estado.h"

/* ---- Pines de camara (Waveshare ESP32-P4-WIFI6) -------------------------- */
#define CAM_SCCB_I2C_PORT   0
#define CAM_SCCB_SCL_PIN    8
#define CAM_SCCB_SDA_PIN    7
#define CAM_SCCB_FREQ_HZ    100000
#define CAM_RESET_PIN       (-1)
#define CAM_PWDN_PIN        (-1)

#define CAM_DEV_PATH        ESP_VIDEO_MIPI_CSI_DEVICE_NAME
#define ISP_DEV_PATH        ESP_VIDEO_ISP1_DEVICE_NAME
#define BUFFER_COUNT        2
#define JPEG_QUALITY        90

/* Discarta unos frames antes de clasificar, para que la tapita recien puesta
 * le de tiempo a la auto-exposicion/foco a reaccionar. El stream nunca se
 * apaga, asi que no hace falta la ventana completa de 6s del ejemplo original. */
#define DISCARD_FRAMES      4

/* Ventana central de muestreo para el color. Frame es 1920x1080. */
#define SAMPLE_WINDOW        300
#define SAMPLE_X0(w)         (((w) - SAMPLE_WINDOW) / 2)
#define SAMPLE_Y0(h)         (((h) - SAMPLE_WINDOW) / 2)

/* Paso de ajuste de ganancia por tecla */
#define WB_GAIN_STEP         0.05f

/* Sensor IR reflectivo FC-51: activo en bajo. En vez de un cooldown por
 * tiempo fijo, se usa una maquina de estados que confirma llegada Y salida
 * -- asi una tapita que tambalea en el lugar (o una mano moviendola para
 * probar) no se cuenta dos veces, pero una tapita nueva que llega despues
 * de que la anterior realmente se fue, si se cuenta.
 *
 * Los dos tiempos de confirmacion son DISTINTOS a proposito. Una tapita
 * rota con agujeros deja pasar la luz por los huecos mientras sigue
 * pasando bajo el sensor -- el sensor "ve" que se destapo un instante,
 * aunque la tapita nunca se fue. Confirmado en la practica: tapitas sanas
 * nunca parpadean asi, las muy rotas si, y por mas de 30ms.
 * Por eso la SALIDA exige mucho mas tiempo sostenido que la LLEGADA:
 * detectar rapido importa, pero declarar "se fue" tiene que ser generoso
 * para no confundir un agujero con que la tapita ya paso. */
#define IR_SENSOR_GPIO        22
#define IR_ARRIVAL_CONFIRM_MS    30    /* filtra ruido electrico al llegar */
#define IR_DEPARTURE_CONFIRM_MS  1000   /* filtra agujeros de tapitas rotas al irse */
#define IR_POLL_INTERVAL_MS      20    /* granularidad del chequeo sostenido */

/* Pines del motor. Test de movimiento: un giro completo (200 pasos a full
 * step, segun lo ya confirmado con el TB6600), a velocidad conservadora
 * para el primer intento. */
#define MOTOR_STEP_GPIO       26
#define MOTOR_DIR_GPIO        54
#define MOTOR_TEST_STEPS       200
#define MOTOR_TEST_STEP_DELAY_US  3000   /* ~166 pasos/seg -> ~1.2s por vuelta */
#define WB_GAIN_STEP_COARSE  0.3f

static const char *TAG = "capture";

static const esp_video_init_csi_config_t csi_config[] = {{
    .sccb_config = {
        .init_sccb = true,
        .i2c_config = { .port = CAM_SCCB_I2C_PORT, .scl_pin = CAM_SCCB_SCL_PIN, .sda_pin = CAM_SCCB_SDA_PIN },
        .freq = CAM_SCCB_FREQ_HZ,
    },
    .reset_pin = CAM_RESET_PIN,
    .pwdn_pin  = CAM_PWDN_PIN,
}};

static const esp_video_init_cam_motor_config_t motor_config[] = {{
    .sccb_config = {
        .init_sccb = true,
        .i2c_config = { .port = CAM_SCCB_I2C_PORT, .scl_pin = CAM_SCCB_SCL_PIN, .sda_pin = CAM_SCCB_SDA_PIN },
        .freq = CAM_SCCB_FREQ_HZ,
    },
    .reset_pin  = -1,
    .pwdn_pin   = -1,
    .signal_pin = -1,
}};

static const esp_video_init_config_t cam_config = {
    .csi = csi_config,
    .cam_motor = motor_config,
};

typedef struct {
    int fd;
    uint8_t *buffer[BUFFER_COUNT];
    size_t buffer_len[BUFFER_COUNT];
    uint32_t width;
    uint32_t height;
} camera_ctx_t;

/* Declaracion adelantada: la usa capture_trigger_task, definida mas abajo
 * en el archivo, antes de la implementacion completa de camera_capture. */
static void camera_capture(camera_ctx_t *ctx, bool send_image);
static void isp_apply_manual_wb(int fd, float red_gain, float blue_gain);

/* fd del nodo ISP (/dev/video20), separado del nodo de captura (/dev/video0) */
static int s_isp_fd = -1;

/* Ganancias manuales actuales, en caliente por teclado */
static float s_red_gain = 1.0f;
static float s_blue_gain = 1.0f;
static bool  s_manual_wb_active = false;

/* Calibradas contra superficie neutra despues del ritual de la luz, con
 * resultado verificado: R-G=-3, B-G=-2, S=0.02 (gris neutro de verdad).
 * Reemplazan a las viejas 1.961/1.283, que correspondian al estado
 * "amarillo" previo al ritual y daban un cast marcado. */
#define CALIBRATED_RED_GAIN    1.603f
#define CALIBRATED_BLUE_GAIN   1.400f

/* Modo streaming: dispara capturas en loop sin esperar comandos, para
 * poder acomodar la camara a mano viendo la imagen actualizarse sola. */
static volatile bool s_streaming = false;

/* Cinta corriendo continua (produccion real: no se frena nunca) */
static volatile bool s_belt_running = false;
#define BELT_STEP_DELAY_US    3000

/* Disparo automatico: sensor detecta -> espera -> captura sola */
static volatile bool s_auto_capture_enabled = false;
static volatile uint32_t s_trigger_delay_ms = 4200;  /* arranca como estimacion, se calibra a ojo */
#define TRIGGER_DELAY_STEP_MS 50

/* Tiempo extra, cinta ya parada, antes de sacar la foto -- para que
 * termine de asentarse cualquier resto de movimiento/vibracion. */
static volatile uint32_t s_settle_delay_ms = 200;
#define SETTLE_DELAY_STEP_MS 50

/* Puntero global al contexto de camara, para que el sensor IR y la cinta
 * puedan disparar una captura sin tener que pasarse el puntero a mano
 * por todos lados. Se setea una sola vez en app_main. */
static camera_ctx_t *s_cam_ctx = NULL;

/* Protege el acceso a los V4L2 ioctls de captura: ahora puede haber varios
 * disparadores (streaming, comando manual, disparo automatico por sensor)
 * y no pueden pisarse llamando DQBUF/QBUF al mismo tiempo. */
static SemaphoreHandle_t s_camera_mutex = NULL;

/* ---------- Motor (TB6600): test de movimiento manual -------------------- */

static void motor_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << MOTOR_STEP_GPIO) | (1ULL << MOTOR_DIR_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(MOTOR_STEP_GPIO, 0);
    gpio_set_level(MOTOR_DIR_GPIO, 0);
    ESP_LOGI(TAG, "Motor inicializado: STEP=GPIO%d DIR=GPIO%d", MOTOR_STEP_GPIO, MOTOR_DIR_GPIO);
}

/* Bit-banging simple: mientras corre, bloquea la tarea que la llama.
 * Para el test manual (unos cientos de ms) esto no es problema. */
static void motor_step(uint32_t steps, bool forward, uint32_t step_delay_us)
{
    /* Invertido: en este cableado, DIR=1 gira hacia atras y DIR=0 hacia
     * adelante -- al reves de lo que uno esperaria por convencion. */
    gpio_set_level(MOTOR_DIR_GPIO, forward ? 0 : 1);
    esp_rom_delay_us(50); /* tiempo de setup entre DIR y el primer pulso de STEP */

    for (uint32_t i = 0; i < steps; i++) {
        gpio_set_level(MOTOR_STEP_GPIO, 1);
        esp_rom_delay_us(step_delay_us / 2);
        gpio_set_level(MOTOR_STEP_GPIO, 0);
        esp_rom_delay_us(step_delay_us / 2);
    }

    ESP_LOGI(TAG, "Motor: %" PRIu32 " pasos %s completados", steps, forward ? "ADELANTE" : "ATRAS");
}

/* Cinta continua: corre mientras s_belt_running este activo. DIR se fija
 * una sola vez al arrancar (0 = adelante, con la inversion ya aplicada).
 *
 * IMPORTANTE: cada tantos pasos hay que cederle la CPU al sistema (con
 * vTaskDelay) o la tarea idle nunca llega a correr y el watchdog reinicia
 * la placa -- esto paso en la practica, no es preventivo de mas. La pausa
 * de ~10ms cada 50 pasos (~150ms) es un hipo imperceptible para la cinta,
 * nada que afecte la clasificacion. */
#define BELT_YIELD_EVERY_STEPS   50

static void belt_task(void *arg)
{
    uint32_t step_count = 0;

    while (1) {
        if (s_belt_running) {
            gpio_set_level(MOTOR_STEP_GPIO, 1);
            esp_rom_delay_us(BELT_STEP_DELAY_US / 2);
            gpio_set_level(MOTOR_STEP_GPIO, 0);
            esp_rom_delay_us(BELT_STEP_DELAY_US / 2);

            step_count++;
            if (step_count % BELT_YIELD_EVERY_STEPS == 0) {
                vTaskDelay(1);
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}

/* ---------- Sensor IR (FC-51): solo deteccion + log, todavia sin ------- */
/* ---------- disparar captura ni mover el motor -------------------------- */

static QueueHandle_t s_ir_evt_queue = NULL;
static QueueHandle_t s_trigger_queue = NULL;

static void IRAM_ATTR ir_isr_handler(void *arg)
{
    uint32_t gpio_num = (uint32_t)(uintptr_t)arg;
    BaseType_t woken = pdFALSE;
    xQueueSendFromISR(s_ir_evt_queue, &gpio_num, &woken);
    if (woken) {
        portYIELD_FROM_ISR();
    }
}

/* Espera a que el pin sostenga 'level_wanted' de forma continua durante
 * 'ms_total', chequeando cada 'IR_POLL_INTERVAL_MS'. Si en algun momento
 * el nivel cambia, se corta y devuelve false -- no alcanza con que el
 * nivel coincida al principio y al final, tiene que sostenerse todo el
 * tramo. Esto es lo que hace falta para no confundir un agujero de una
 * tapita rota (un parpadeo a mitad de camino) con una salida real. */
static bool ir_level_held(int level_wanted, uint32_t ms_total)
{
    uint32_t esperado = 0;
    while (esperado < ms_total) {
        uint32_t paso = (ms_total - esperado < IR_POLL_INTERVAL_MS)
        ? (ms_total - esperado) : IR_POLL_INTERVAL_MS;
        vTaskDelay(pdMS_TO_TICKS(paso));
        esperado += paso;
        if (gpio_get_level(IR_SENSOR_GPIO) != level_wanted) {
            return false;
        }
    }
    return true;
}

static void ir_sensor_task(void *arg)
{
    uint32_t gpio_num;
    typedef enum { IR_STATE_LIBRE, IR_STATE_OCUPADO } ir_state_t;
    ir_state_t state = IR_STATE_LIBRE;
    uint32_t count = 0;

    while (1) {
        if (xQueueReceive(s_ir_evt_queue, &gpio_num, portMAX_DELAY)) {
            int level = gpio_get_level(IR_SENSOR_GPIO);  /* 0 = objeto presente (activo en bajo) */

            if (state == IR_STATE_LIBRE && level == 0) {
                /* posible llegada: confirmar que se sostiene, no es ruido */
                if (ir_level_held(0, IR_ARRIVAL_CONFIRM_MS)) {
                    count++;
                    state = IR_STATE_OCUPADO;
                    ESP_LOGI(TAG, "===== SENSOR IR: deteccion #%" PRIu32 " (GPIO%d) =====",
                             count, (int)gpio_num);
                    if (s_auto_capture_enabled && s_trigger_queue) {
                        xQueueSend(s_trigger_queue, &count, 0);
                    }
                }
            } else if (state == IR_STATE_OCUPADO && level == 1) {
                /* posible salida: exigir que se sostenga destapado bastante
                 * mas tiempo, para no confundir un agujero con que ya se fue */
                if (ir_level_held(1, IR_DEPARTURE_CONFIRM_MS)) {
                    state = IR_STATE_LIBRE;
                }
            }
            /* cualquier otro caso (parpadeo dentro del mismo estado) se ignora */
        }
    }
}

static void ir_sensor_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << IR_SENSOR_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,    /* activo en bajo */
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_ANYEDGE,      /* necesitamos saber tanto cuando llega como cuando se va */
    };
    gpio_config(&io_conf);

    s_ir_evt_queue = xQueueCreate(10, sizeof(uint32_t));
    s_trigger_queue = xQueueCreate(10, sizeof(uint32_t));
    gpio_install_isr_service(0);
    gpio_isr_handler_add(IR_SENSOR_GPIO, ir_isr_handler, (void *)(uintptr_t)IR_SENSOR_GPIO);

    xTaskCreate(ir_sensor_task, "ir_sensor_task", 3072, NULL, 10, NULL);
    ESP_LOGI(TAG, "Sensor IR inicializado en GPIO%d (llegada %dms, salida %dms, por presencia real)",
             IR_SENSOR_GPIO, IR_ARRIVAL_CONFIRM_MS, IR_DEPARTURE_CONFIRM_MS);
}

/* Espera el tiempo que tarda la tapita en viajar del sensor a la camara
 * (s_trigger_delay_ms, calibrable en caliente con 'y'/'Y') y dispara la
 * captura sola, sin que nadie toque una tecla. */
static void capture_trigger_task(void *arg)
{
    uint32_t id;
    while (1) {
        if (xQueueReceive(s_trigger_queue, &id, portMAX_DELAY)) {
            /* Fase 1: dejar que la cinta avance el tiempo ya calibrado,
             * hasta que la tapita quede debajo de la camara. */
            vTaskDelay(pdMS_TO_TICKS(s_trigger_delay_ms));

            /* Fase 2: frenar, dejar asentar, sacar la foto quieta. */
            bool was_running = s_belt_running;
            s_belt_running = false;
            vTaskDelay(pdMS_TO_TICKS(s_settle_delay_ms));

            if (s_cam_ctx) {
                ESP_LOGI(TAG, "Disparo automatico #%" PRIu32 " (viaje=%" PRIu32 "ms, asentado=%" PRIu32 "ms, cinta parada)",
                         id, s_trigger_delay_ms, s_settle_delay_ms);
                camera_capture(s_cam_ctx, true);
            }

            /* Fase 3: retomar la cinta para sacar la tapita clasificada
             * y dejar lugar a la proxima. */
            if (was_running) {
                s_belt_running = true;
            }
        }
    }
}

/* Variante de 'f': fija la ganancia de WB pero NO apaga las estadisticas.
 * Hipotesis: apagar las stats AWB frena todo el algoritmo del ISP (incluida
 * la auto-exposicion), y por eso despues la imagen no se re-expone al
 * cambiar de tapita y satura. Si el WB manual se sostiene igual con las
 * stats corriendo, ganamos las dos cosas: color fijo + exposicion viva. */
static void apply_wb_keeping_stats(void)
{
    s_red_gain = CALIBRATED_RED_GAIN;
    s_blue_gain = CALIBRATED_BLUE_GAIN;
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
    s_manual_wb_active = true;
    ESP_LOGI(TAG, "WB fijo aplicado SIN apagar las estadisticas (auto-exposicion deberia seguir viva).");
    ESP_LOGI(TAG, "Chequeo: sacale foto a la amarilla y a la blanca -- si el RGB ya no queda clavado "
    "en 255,255,0 y cambia entre tapitas, la teoria era correcta.");
}

/* ---------- Ventana de estadisticas: que AE y AF miren la tapita -------- */
/* El AE por defecto promedia TODO el cuadro. Como el fondo oscuro ocupa
 * mucha mas area que la tapita, concluye "esto esta oscuro" y sube la
 * exposicion hasta quemar la tapita (de ahi el clipping a 255).
 * Restringiendo la ventana al centro, expone para lo que importa.
 * Se aplica tambien a AF para que enfoque sobre la tapita, no sobre el fondo.
 * NO se toca la ventana de AWB a proposito: el balance de blancos tiene que
 * seguir midiendo la escena completa, si midiera solo la tapita de color
 * intentaria "neutralizarla" y arruinaria el color. */

#define STATS_WINDOW_SIZE   700   /* mas grande que SAMPLE_WINDOW, cubre la tapita entera */

static void set_center_stats_window(uint32_t frame_w, uint32_t frame_h)
{
    uint32_t left = (frame_w - STATS_WINDOW_SIZE) / 2;
    uint32_t top  = (frame_h > STATS_WINDOW_SIZE) ? (frame_h - STATS_WINDOW_SIZE) / 2 : 0;
    uint32_t w = STATS_WINDOW_SIZE;
    uint32_t h = (frame_h > STATS_WINDOW_SIZE) ? STATS_WINDOW_SIZE : frame_h;

    esp_err_t err = esp_video_isp_pipeline_set_statistics_window(
        ESP_VIDEO_ISP_AE_STATS_WIN | ESP_VIDEO_ISP_AF_STATS_WIN,
        left, top, w, h);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "===== Ventana de estadisticas AE+AF centrada: %" PRIu32 "x%" PRIu32
        " en (%" PRIu32 ",%" PRIu32 ") =====", w, h, left, top);
        ESP_LOGI(TAG, "Ahora la exposicion y el foco se calculan sobre la tapita, no sobre el fondo.");
    } else {
        ESP_LOGE(TAG, "No pude fijar la ventana de estadisticas: %s", esp_err_to_name(err));
    }
}

/* Diagnostico: poner el WB en ganancia neutra (1.0/1.0) sin apagar stats.
 * Si con esto la tapita amarilla deja de clipear, confirma que el clipping
 * lo causan nuestras ganancias de correccion, no la exposicion. */
static void apply_neutral_wb(void)
{
    s_red_gain = 1.0f;
    s_blue_gain = 1.0f;
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
    s_manual_wb_active = true;
    ESP_LOGI(TAG, "WB NEUTRO (1.0/1.0) aplicado -- solo para diagnostico de clipping.");
    ESP_LOGI(TAG, "Los colores van a verse mal (sin corregir), pero si el amarillo ya no da 255,255,x "
    "entonces el clipping lo causaban las ganancias.");
}

static void show_agc_status(void)
{
    esp_video_isp_pipeline_agc_status_t status;
    if (esp_video_isp_pipeline_get_agc_status(&status) == ESP_OK) {
        ESP_LOGI(TAG, "AGC (auto-ganancia/exposicion): %s",
                 (status == ESP_VIDEO_ISP_PIPELINE_AGC_ENABLE) ? "HABILITADO" : "DESHABILITADO");
        if (status != ESP_VIDEO_ISP_PIPELINE_AGC_ENABLE) {
            ESP_LOGW(TAG, "Con el AGC deshabilitado, cambiar el techo de exposicion no hace nada. "
            "Eso explicaria por que 'w' no movia la imagen.");
        }
    } else {
        ESP_LOGE(TAG, "No pude leer el estado del AGC.");
    }
}

/* ---------- Techo de exposicion via el ISP pipeline controller ----------- */
/* Esta API existe solo si CONFIG_ESP_VIDEO_ENABLE_ISP_PIPELINE_CONTROLLER
 * esta activado en menuconfig (confirmado que si lo esta). Es la forma
 * limpia de evitar el clipping: en vez de bajar la luz fisica, se le pone
 * un techo al tiempo que el sensor deja entrar luz. */

static uint32_t s_max_exposure_us = 0;   /* 0 = todavia no leido */
#define EXPOSURE_STEP_FACTOR   0.75f

static void show_exposure_limits(void)
{
    uint32_t max_us = 0, min_us = 0;
    esp_err_t r1 = esp_video_isp_pipeline_get_agc_max_exposure(&max_us);
    esp_err_t r2 = esp_video_isp_pipeline_get_agc_min_exposure(&min_us);

    if (r1 == ESP_OK && r2 == ESP_OK) {
        s_max_exposure_us = max_us;
        ESP_LOGI(TAG, "===== Exposicion AGC: min=%" PRIu32 "us  max=%" PRIu32 "us =====",
                 min_us, max_us);
        ESP_LOGI(TAG, "Bajar el max con 'w' reduce el clipping (menos luz entrando al sensor).");
    } else {
        ESP_LOGE(TAG, "No pude leer los limites de exposicion (max=%s, min=%s)",
                 esp_err_to_name(r1), esp_err_to_name(r2));
    }
}

static void adjust_max_exposure(float factor)
{
    if (s_max_exposure_us == 0) {
        uint32_t max_us = 0;
        if (esp_video_isp_pipeline_get_agc_max_exposure(&max_us) != ESP_OK) {
            ESP_LOGE(TAG, "No pude leer la exposicion maxima actual.");
            return;
        }
        s_max_exposure_us = max_us;
    }

    uint32_t nuevo = (uint32_t)(s_max_exposure_us * factor);
    if (nuevo < 1) nuevo = 1;

    esp_err_t err = esp_video_isp_pipeline_set_agc_max_exposure(nuevo);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Exposicion maxima: %" PRIu32 "us -> %" PRIu32 "us", s_max_exposure_us, nuevo);
        s_max_exposure_us = nuevo;
    } else {
        ESP_LOGE(TAG, "No pude fijar exposicion maxima en %" PRIu32 "us: %s "
        "(puede estar fuera del rango que acepta el sensor)",
                 nuevo, esp_err_to_name(err));
    }
}

/* Apagar el autofoco continuo. Sin esto, fijar el VCM a mano no sirve:
 * el algoritmo lo vuelve a mover en cuanto cambia la escena (verificado:
 * al sacar la tapita, el foco se iba solo). Se prueban dos vias porque
 * no esta documentado cual soporta este driver. */
static void disable_autofocus(int cam_fd, int isp_fd)
{
    bool ok = false;

    /* Via 1: control estandar de V4L2 en el nodo de camara. */
    struct v4l2_control af = { .id = V4L2_CID_FOCUS_AUTO, .value = 0 };
    if (ioctl(cam_fd, VIDIOC_S_CTRL, &af) == 0) {
        ESP_LOGI(TAG, "Autofoco apagado via V4L2_CID_FOCUS_AUTO.");
        ok = true;
    } else {
        ESP_LOGW(TAG, "V4L2_CID_FOCUS_AUTO no soportado (errno=%d), probando por el ISP...", errno);
    }

    /* Via 2: apagar las estadisticas de AF en el ISP. Sin datos, el
     * algoritmo de enfoque no tiene con que decidir y deja de mover el VCM. */
    esp_video_isp_af_t af_cfg = { .enable = false };
    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_AF,
        .size = sizeof(af_cfg),
        .ptr = &af_cfg,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };
    if (ioctl(isp_fd, VIDIOC_S_EXT_CTRLS, &ctrls) == 0) {
        ESP_LOGI(TAG, "Estadisticas de AF desactivadas en el ISP.");
        ok = true;
    } else {
        ESP_LOGW(TAG, "No pude desactivar las stats de AF (errno=%d %s)", errno, strerror(errno));
    }

    if (ok) {
        ESP_LOGI(TAG, "===== AUTOFOCO APAGADO. Ahora 'o'/'O' fijan el foco de verdad. =====");
    } else {
        ESP_LOGE(TAG, "No pude apagar el autofoco por ninguna via.");
    }
}

/* Foco fijo calibrado: la distancia camara-tapita es constante, asi que el
 * autofoco no aporta nada y solo agrega demora (~8s) y movimiento del VCM.
 * 886 es el valor donde enfoca bien, medido con tapita en posicion. */
#define CALIBRATED_FOCUS   886

/* ---------- Foco manual: fijar el VCM en una posicion conocida ---------- */
/* En este sistema la distancia camara-tapita es constante, asi que el
 * autofoco continuo no aporta nada y solo puede meter ruido. Estos
 * comandos permiten clavarlo donde enfoque bien y dejarlo ahi. */

static int s_manual_focus = -1;
#define FOCUS_STEP   20

static void set_manual_focus(int fd, int pos)
{
    if (pos < 0) pos = 0;
    if (pos > 1023) pos = 1023;   /* rango del DW9807 segun el log de boot */

        struct v4l2_ext_control ctrl = { .id = V4L2_CID_FOCUS_ABSOLUTE, .value = pos };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_CAMERA_CLASS, .count = 1, .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) == 0) {
        s_manual_focus = pos;
        ESP_LOGI(TAG, "===== FOCO MANUAL fijado en %d =====", pos);
    } else {
        ESP_LOGE(TAG, "No pude fijar el foco en %d (errno=%d %s)", pos, errno, strerror(errno));
    }
}

static void adjust_manual_focus(int fd, int delta)
{
    if (s_manual_focus < 0) {
        /* Primera vez: apagar el autofoco antes de nada, o va a pisar
         * cualquier valor que fijemos. */
        disable_autofocus(fd, s_isp_fd);

        /* Partir de donde este ahora el autofoco, no de cero */
        struct v4l2_ext_control ctrl = { .id = V4L2_CID_FOCUS_ABSOLUTE };
        struct v4l2_ext_controls ctrls = {
            .ctrl_class = V4L2_CID_CAMERA_CLASS, .count = 1, .controls = &ctrl,
        };
        s_manual_focus = (ioctl(fd, VIDIOC_G_EXT_CTRLS, &ctrls) == 0) ? ctrl.value : 512;
        ESP_LOGI(TAG, "Tomando el foco actual (%d) como punto de partida.", s_manual_focus);
    }
    set_manual_focus(fd, s_manual_focus + delta);
}

/* Bajar el PISO de exposicion. Si el AGC ya esta trabajando contra el
 * minimo (8333us por defecto) y la imagen igual satura, mover el techo no
 * sirve de nada -- el limite que muerde es el de abajo. Se bajan los dos
 * juntos, manteniendo min <= max, para que el AGC tenga lugar donde bajar. */
static void adjust_exposure_floor(float factor)
{
    uint32_t min_us = 0, max_us = 0;
    if (esp_video_isp_pipeline_get_agc_min_exposure(&min_us) != ESP_OK ||
        esp_video_isp_pipeline_get_agc_max_exposure(&max_us) != ESP_OK) {
        ESP_LOGE(TAG, "No pude leer los limites actuales de exposicion.");
    return;
        }

        uint32_t nuevo_min = (uint32_t)(min_us * factor);
        if (nuevo_min < 1) nuevo_min = 1;

        /* El techo tiene que quedar por encima del piso; si el piso baja mucho,
         * se baja el techo tambien para que el AGC pueda moverse en ese rango. */
        uint32_t nuevo_max = max_us;
        if (nuevo_max > nuevo_min * 4) {
            nuevo_max = nuevo_min * 4;
        }
        if (nuevo_max < nuevo_min) nuevo_max = nuevo_min;

        esp_err_t e1 = esp_video_isp_pipeline_set_agc_min_exposure(nuevo_min);
    esp_err_t e2 = esp_video_isp_pipeline_set_agc_max_exposure(nuevo_max);

    ESP_LOGI(TAG, "Exposicion: min %" PRIu32 "->%" PRIu32 "us (%s)  max %" PRIu32 "->%" PRIu32 "us (%s)",
             min_us, nuevo_min, esp_err_to_name(e1),
             max_us, nuevo_max, esp_err_to_name(e2));

    if (e1 != ESP_OK) {
        ESP_LOGW(TAG, "El sensor rechazo ese minimo -- puede que %" PRIu32 "us sea mas corto "
        "de lo que soporta a esta resolucion.", nuevo_min);
    }
    s_max_exposure_us = nuevo_max;
}

/* Congelar la exposicion. Mismo criterio que con el balance de blancos:
 * se deja que el AGC converja con la tapita bien puesta, y despues se lo
 * apaga para que ese valor quede fijo. Sin esto, mover la cinta hace que
 * el AE vea el fondo oscuro, suba la exposicion, y no vuelva a bajar --
 * verificado en la practica: la misma tapita naranja pasaba de G=123
 * (H=30, correcto) a G=214 (H=51, se lee como amarillo) despues de
 * mover el motor. */
static void freeze_exposure(void)
{
    esp_err_t err = esp_video_isp_pipeline_set_agc_status(ESP_VIDEO_ISP_PIPELINE_AGC_DISABLE);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "===== EXPOSICION CONGELADA (AGC apagado) =====");
        ESP_LOGI(TAG, "Ya no se va a re-ajustar sola al mover la cinta. Si la luz cambia, "
        "apreta 'D' para descongelar, dejala converger, y volve a congelar con 'd'.");
    } else {
        ESP_LOGE(TAG, "No pude congelar la exposicion: %s", esp_err_to_name(err));
    }
}

static void unfreeze_exposure(void)
{
    esp_err_t err = esp_video_isp_pipeline_set_agc_status(ESP_VIDEO_ISP_PIPELINE_AGC_ENABLE);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "===== EXPOSICION AUTOMATICA REACTIVADA =====");
        ESP_LOGI(TAG, "Deja que se asiente unos segundos con la tapita puesta, y congela con 'd'.");
    } else {
        ESP_LOGE(TAG, "No pude reactivar el AGC: %s", esp_err_to_name(err));
    }
}

/* Exposicion FIJA en un valor exacto. Poniendo piso y techo en el mismo
 * numero, el AGC no tiene margen para moverse: todos los arranques dan
 * el mismo resultado, sin depender de que hubiera en el cuadro al
 * congelar. Esto es lo que hacia que los umbrales no se sostuvieran
 * entre sesiones aun con el AGC apagado. */
static uint32_t s_fixed_exposure_us = 6000;   /* punto de partida, se ajusta con '1'/'2' */

static void set_fixed_exposure(uint32_t us)
{
    if (us < 100) us = 100;

    /* Orden importante: primero bajar el piso, despues el techo, para no
     * pasar por un estado invalido donde min > max. */
    esp_err_t e1 = esp_video_isp_pipeline_set_agc_min_exposure(us);
    esp_err_t e2 = esp_video_isp_pipeline_set_agc_max_exposure(us);

    if (e1 == ESP_OK && e2 == ESP_OK) {
        s_fixed_exposure_us = us;
        ESP_LOGI(TAG, "===== EXPOSICION FIJA en %" PRIu32 "us (min=max, sin margen) =====", us);
    } else {
        ESP_LOGE(TAG, "No pude fijar la exposicion en %" PRIu32 "us (min:%s max:%s)",
                 us, esp_err_to_name(e1), esp_err_to_name(e2));
        ESP_LOGW(TAG, "Si el sensor rechaza el valor, probá uno mas alto con '2'.");
    }

    uint32_t min_leido = 0, max_leido = 0;
    esp_video_isp_pipeline_get_agc_min_exposure(&min_leido);
    esp_video_isp_pipeline_get_agc_max_exposure(&max_leido);
    ESP_LOGI(TAG, "  verificacion: min=%" PRIu32 "us max=%" PRIu32 "us", min_leido, max_leido);
}

/* ---------- Exposicion/ganancia: sondeo y ajuste ------------------------- */

typedef struct { uint32_t id; const char *name; } known_ctrl_t;

static const known_ctrl_t s_exposure_ctrls[] = {
    { V4L2_CID_EXPOSURE,           "EXPOSURE" },
    { V4L2_CID_EXPOSURE_ABSOLUTE,  "EXPOSURE_ABSOLUTE" },
    { V4L2_CID_GAIN,               "GAIN" },
    #ifdef V4L2_CID_ANALOGUE_GAIN
    { V4L2_CID_ANALOGUE_GAIN,      "ANALOGUE_GAIN" },
    #endif
    { V4L2_CID_BRIGHTNESS,         "BRIGHTNESS" },
};

/* Sondea por ID directo (la enumeracion con NEXT_CTRL no anda en este
 * driver). Muestra rango y valor actual de lo que si este soportado. */
static void probe_exposure_controls(int fd)
{
    ESP_LOGI(TAG, "===== Sondeo de controles de exposicion/ganancia =====");
    for (size_t i = 0; i < sizeof(s_exposure_ctrls) / sizeof(s_exposure_ctrls[0]); i++) {
        struct v4l2_queryctrl qc = { .id = s_exposure_ctrls[i].id };
        if (ioctl(fd, VIDIOC_QUERYCTRL, &qc) == 0) {
            struct v4l2_control cur = { .id = s_exposure_ctrls[i].id };
            int have_cur = (ioctl(fd, VIDIOC_G_CTRL, &cur) == 0);
            ESP_LOGI(TAG, "  [OK] %s: min=%" PRId32 " max=%" PRId32 " step=%" PRId32
            " default=%" PRId32 " actual=%s%" PRId32,
            s_exposure_ctrls[i].name, (int32_t)qc.minimum, (int32_t)qc.maximum,
                     (int32_t)qc.step, (int32_t)qc.default_value,
                     have_cur ? "" : "(no leible) ", (int32_t)(have_cur ? cur.value : 0));
        } else {
            ESP_LOGW(TAG, "  [no soportado] %s (errno=%d)", s_exposure_ctrls[i].name, errno);
        }
    }
    ESP_LOGI(TAG, "===== fin del sondeo =====");
}

/* Ajuste relativo del primer control de exposicion que este soportado. */
static void adjust_exposure(int fd, float factor)
{
    for (size_t i = 0; i < sizeof(s_exposure_ctrls) / sizeof(s_exposure_ctrls[0]); i++) {
        struct v4l2_queryctrl qc = { .id = s_exposure_ctrls[i].id };
        if (ioctl(fd, VIDIOC_QUERYCTRL, &qc) != 0) {
            continue;
        }
        struct v4l2_control cur = { .id = s_exposure_ctrls[i].id };
        if (ioctl(fd, VIDIOC_G_CTRL, &cur) != 0) {
            continue;
        }

        int32_t nuevo = (int32_t)(cur.value * factor);
        if (nuevo == cur.value) {  /* que un factor chico igual mueva algo */
            nuevo = (factor < 1.0f) ? cur.value - 1 : cur.value + 1;
        }
        if (nuevo < qc.minimum) nuevo = qc.minimum;
        if (nuevo > qc.maximum) nuevo = qc.maximum;

        struct v4l2_control set = { .id = s_exposure_ctrls[i].id, .value = nuevo };
        if (ioctl(fd, VIDIOC_S_CTRL, &set) == 0) {
            ESP_LOGI(TAG, "%s: %" PRId32 " -> %" PRId32 " (rango %" PRId32 "..%" PRId32 ")",
                     s_exposure_ctrls[i].name, (int32_t)cur.value, nuevo,
                     (int32_t)qc.minimum, (int32_t)qc.maximum);
        } else {
            ESP_LOGE(TAG, "%s: no pude escribir (errno=%d %s)",
                     s_exposure_ctrls[i].name, errno, strerror(errno));
        }
        return;
    }
    ESP_LOGW(TAG, "Ningun control de exposicion soportado -- apreta 'x' para ver el sondeo completo.");
}

/* ---------- Puente por serial: la clasificacion viaja por el mismo cable -- */
/* que ya usan las imagenes, nunca fallo en toda la sesion. El WiFi/MQTT
 * directo desde el ESP32 se saco por completo: es la causa confirmada de
 * los reinicios sin fondo ("Unrecoverable host sdio state", bug conocido
 * y sin resolver de esp-hosted-mcu para P4+C6). El puente lo arma
 * ver_imx708.py en la PC, que lee esta linea y la publica el mismo por
 * MQTT a la Raspberry -- el ESP32 no vuelve a tocar el C6 para nada. */

static char s_sesion_id[16] = {0};

/* Ultima clasificacion calculada, la escribe classify_and_log(). */
static const char *s_last_color = "";
static float s_last_h = 0, s_last_s = 0, s_last_v = 0;

/* Ultima probabilidad de "rota" calculada por el modelo TinyML, la escribe
 * camera_capture() despues de classify_and_log(). Se manda la probabilidad
 * CRUDA (0.0-1.0), sin aplicar umbral aca -- la decision sana/rota se toma
 * rio abajo en tapitas_ingest.py (Raspberry), asi se puede ajustar el
 * umbral sin reflashear el ESP32. -1.0 = todavia no se corrio el modelo
 * (falla de init o primera captura antes de tinyml_init). */
static float s_last_prob_rota = -1.0f;

static void publish_clasificacion_por_serial(void)
{
    /* Prefijo "MQTTDATA " para que el script en la PC lo reconozca sin
     * confundirlo con cualquier otra linea de log normal. */
    printf("MQTTDATA {\"color\":\"%s\",\"hue\":%.2f,\"saturation\":%.3f,"
    "\"value\":%.3f,\"prob_rota\":%.4f,\"sesion_id\":\"%s\"}\n",
    s_last_color, s_last_h, s_last_s, s_last_v, s_last_prob_rota, s_sesion_id);
}

/* ---------- Diagnostico: listar controles V4L2 disponibles --------------- */

static void list_available_controls(int fd, const char *dev_name)
{
    ESP_LOGI(TAG, "===== Controles V4L2 disponibles en %s =====", dev_name);
    struct v4l2_queryctrl qc = {0};
    qc.id = V4L2_CTRL_FLAG_NEXT_CTRL;
    int found = 0;

    while (ioctl(fd, VIDIOC_QUERYCTRL, &qc) == 0) {
        found++;
        ESP_LOGI(TAG, "  id=0x%08" PRIx32 " name='%s' type=%" PRIu32,
                 (uint32_t)qc.id, qc.name, (uint32_t)qc.type);
        qc.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }

    if (found == 0) {
        ESP_LOGW(TAG, "  VIDIOC_QUERYCTRL con NEXT_CTRL no devolvio nada (errno=%d %s). "
        "Normal para controles extendidos tipo array (WB/AWB/CCM, etc): "
        "esos no se enumeran asi, se usan directo por ID con S/G_EXT_CTRLS.",
        errno, strerror(errno));
    }
    ESP_LOGI(TAG, "===== fin de la lista =====");
}

/* ---------- Balance de blancos manual, via el nodo ISP -------------------- */

/* Config AWB de fabrica (la que trae el imx708_default.json al bootear),
 * guardada ANTES de tocar nada. Reactivar el AWB con enable=true a secas
 * (todo lo demas en cero) lo rechaza el hardware -- necesita rangos
 * validos de verdad, y estos son los unicos que se sabe que el sensor
 * acepta, porque son los que el mismo driver ya tenia cargados. */
static esp_video_isp_awb_t s_default_awb_config = {0};
static bool s_have_default_awb_config = false;

static void isp_save_default_awb_config(int fd)
{
    esp_video_isp_awb_t awb = {0};
    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_AWB,
        .size = sizeof(awb),
        .ptr = &awb,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_G_EXT_CTRLS, &ctrls) == 0) {
        s_default_awb_config = awb;
        s_have_default_awb_config = true;
        ESP_LOGI(TAG, "Config AWB de fabrica guardada (green %u-%u, rg %.3f-%.3f, bg %.3f-%.3f)",
                 awb.green_min, awb.green_max, awb.rg_min, awb.rg_max, awb.bg_min, awb.bg_max);
    } else {
        ESP_LOGW(TAG, "No pude leer la config AWB de fabrica (errno=%d %s) -- 'n' podria fallar mas adelante",
                 errno, strerror(errno));
    }
}

static void isp_disable_awb_stats(int fd)
{
    esp_video_isp_awb_t awb = { .enable = false };
    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_AWB,
        .size = sizeof(awb),
        .ptr = &awb,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGE(TAG, "No pude desactivar estadisticas AWB (errno=%d %s)",
                 errno, strerror(errno));
    } else {
        ESP_LOGI(TAG, "Estadisticas AWB (algoritmo automatico) desactivadas.");
    }
}

/* Contraparte: reactiva el AWB automatico para que pueda re-converger bajo
 * la luz real del momento. Necesario para poder recalibrar con 'm' --
 * si el AWB nunca corre, 'm' solo puede devolver el ultimo valor fijado. */
static void isp_enable_awb_stats(int fd)
{
    esp_video_isp_awb_t awb;
    if (s_have_default_awb_config) {
        awb = s_default_awb_config;
        awb.enable = true;
    } else {
        ESP_LOGW(TAG, "No tengo guardada la config de fabrica -- probando con enable=true a secas, puede fallar.");
        awb = (esp_video_isp_awb_t){ .enable = true };
    }

    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_AWB,
        .size = sizeof(awb),
        .ptr = &awb,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGE(TAG, "No pude reactivar estadisticas AWB (errno=%d %s)",
                 errno, strerror(errno));
        return;
    }

    /* CLAVE: soltar tambien el WB manual. Reactivar las estadisticas no
     * alcanza -- si el WB sigue en enable=true con ganancia fija, el
     * algoritmo puede medir pero no puede aplicar nada, y la imagen queda
     * clavada en la ganancia vieja para siempre. */
    esp_video_isp_wb_t wb = { .enable = false };
    struct v4l2_ext_control wb_ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_WB,
        .size = sizeof(wb),
        .ptr = &wb,
    };
    struct v4l2_ext_controls wb_ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &wb_ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &wb_ctrls) != 0) {
        ESP_LOGE(TAG, "Reactive las estadisticas pero no pude soltar el WB manual (errno=%d %s) "
        "-- el AWB va a medir pero no va a poder corregir nada.",
        errno, strerror(errno));
        return;
    }

    s_manual_wb_active = false;
    ESP_LOGI(TAG, "===== AWB AUTOMATICO REACTIVADO (control devuelto de verdad). =====");
    ESP_LOGI(TAG, "Ahora hace el ritual de la luz: subi el brillo hasta que se vea bien clara, "
    "bajalo de nuevo, y cuando la imagen te guste apreta 'm' para congelar ese valor.");
}

static void isp_apply_manual_wb(int fd, float red_gain, float blue_gain)
{
    esp_video_isp_wb_t wb = {
        .enable = true,
        .red_gain = red_gain,
        .blue_gain = blue_gain,
    };
    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_WB,
        .size = sizeof(wb),
        .ptr = &wb,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGE(TAG, "No pude fijar WB manual (errno=%d %s)", errno, strerror(errno));
    } else {
        ESP_LOGI(TAG, "===== WB MANUAL: red_gain=%.3f blue_gain=%.3f =====", red_gain, blue_gain);
    }
}

static bool isp_read_current_wb(int fd, float *red_gain, float *blue_gain)
{
    esp_video_isp_wb_t wb = {0};
    struct v4l2_ext_control ctrl = {
        .id = V4L2_CID_USER_ESP_ISP_WB,
        .size = sizeof(wb),
        .ptr = &wb,
    };
    struct v4l2_ext_controls ctrls = {
        .ctrl_class = V4L2_CID_USER_CLASS,
        .count = 1,
        .controls = &ctrl,
    };

    if (ioctl(fd, VIDIOC_G_EXT_CTRLS, &ctrls) != 0) {
        ESP_LOGE(TAG, "No pude leer WB actual (errno=%d %s)", errno, strerror(errno));
        return false;
    }

    ESP_LOGI(TAG, "WB leido (mientras el AWB automatico corria): enable=%d red_gain=%.3f blue_gain=%.3f",
             wb.enable, wb.red_gain, wb.blue_gain);
    *red_gain = wb.red_gain;
    *blue_gain = wb.blue_gain;
    return true;
}

static void enable_manual_white_balance(void)
{
    /* CLAVE: leer la ganancia que el AWB automatico ya habia convergido
     * ANTES de apagarlo. Si arrancamos en 1.0/1.0 a ciegas, deshacemos
     * toda la correccion que el algoritmo ya habia logrado y la imagen
     * pega un salto de color (visto en la practica: blanco correcto ->
     * verde fuerte al activar 'm' con ganancia neutra). */
    float auto_red = 1.0f, auto_blue = 1.0f;
    if (isp_read_current_wb(s_isp_fd, &auto_red, &auto_blue)) {
        s_red_gain = auto_red;
        s_blue_gain = auto_blue;
        ESP_LOGI(TAG, "Arrancando WB manual desde la ganancia ya convergida por el AWB (no desde 1.0/1.0).");
    } else {
        ESP_LOGW(TAG, "No pude leer la ganancia automatica -- arranco en 1.0/1.0, puede pegar un salto de color.");
    }

    isp_disable_awb_stats(s_isp_fd);
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
    s_manual_wb_active = true;
    ESP_LOGI(TAG, "WB manual activo. 'r'/'R' = red_gain -/+  'b'/'B' = blue_gain -/+  'p' = ver valores actuales");
}

static void adjust_red_gain(float delta)
{
    if (!s_manual_wb_active) {
        ESP_LOGW(TAG, "WB manual no esta activo todavia. Apreta 'm' primero.");
        return;
    }
    s_red_gain += delta;
    if (s_red_gain < 0.1f) s_red_gain = 0.1f;
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
}

static void adjust_blue_gain(float delta)
{
    if (!s_manual_wb_active) {
        ESP_LOGW(TAG, "WB manual no esta activo todavia. Apreta 'm' primero.");
        return;
    }
    s_blue_gain += delta;
    if (s_blue_gain < 0.1f) s_blue_gain = 0.1f;
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
}

static void print_current_wb(void)
{
    ESP_LOGI(TAG, "WB manual: %s | red_gain=%.3f blue_gain=%.3f",
             s_manual_wb_active ? "ACTIVO" : "inactivo (AWB automatico corriendo)",
             s_red_gain, s_blue_gain);
}

/* ---------- Clasificacion de color (misma logica que la OV5647) ---------- */

/* Solo cuentan los pixeles con brillo (max de R/G/B) >= DARK_PIXEL_RATIO del
 * brillo medio de la ventana. Los agujeros de una tapita rota dejan ver el
 * fondo oscuro, que tiraba el matiz promedio hacia el rojo (una naranja rota
 * medio H=13 contra H=22 de una sana, misma luz). */
#define DARK_PIXEL_RATIO     0.70f

static void sample_average_rgb_region(const uint8_t *frame, uint32_t frame_w,
                                      int x0, int y0, int w, int h,
                                      float *r_out, float *g_out, float *b_out)
{
    uint64_t sum_max = 0;
    uint32_t total = 0;

    for (int y = y0; y < y0 + h; y++) {
        const uint16_t *row = (const uint16_t *)(frame + (size_t)y * frame_w * 2);
        for (int x = x0; x < x0 + w; x++) {
            uint16_t p = row[x];
            uint32_t r = (((p >> 11) & 0x1F) * 255) / 31;
            uint32_t g = (((p >> 5) & 0x3F) * 255) / 63;
            uint32_t b = ((p & 0x1F) * 255) / 31;
            uint32_t m = r > g ? (r > b ? r : b) : (g > b ? g : b);
            sum_max += m;
            total++;
        }
    }
    uint32_t min_max = (uint32_t)(DARK_PIXEL_RATIO * (float)sum_max / total);

    uint64_t sum_r = 0, sum_g = 0, sum_b = 0;
    uint32_t count = 0;

    for (int y = y0; y < y0 + h; y++) {
        const uint16_t *row = (const uint16_t *)(frame + (size_t)y * frame_w * 2);
        for (int x = x0; x < x0 + w; x++) {
            uint16_t p = row[x];
            uint32_t r = (((p >> 11) & 0x1F) * 255) / 31;
            uint32_t g = (((p >> 5) & 0x3F) * 255) / 63;
            uint32_t b = ((p & 0x1F) * 255) / 31;
            uint32_t m = r > g ? (r > b ? r : b) : (g > b ? g : b);
            if (m < min_max) continue;
            sum_r += r;
            sum_g += g;
            sum_b += b;
            count++;
        }
    }

    ESP_LOGI(TAG, "color: %" PRIu32 "/%" PRIu32 " pixeles usados (se descartan los oscuros)", count, total);
    *r_out = (float)sum_r / count;
    *g_out = (float)sum_g / count;
    *b_out = (float)sum_b / count;
}

static void rgb_to_hsv(float r, float g, float b, float *h, float *s, float *v)
{
    r /= 255.0f; g /= 255.0f; b /= 255.0f;
    float maxc = fmaxf(r, fmaxf(g, b));
    float minc = fminf(r, fminf(g, b));
    float delta = maxc - minc;

    *v = maxc;
    *s = (maxc <= 0.0001f) ? 0.0f : (delta / maxc);

    if (delta <= 0.0001f) {
        *h = 0;
    } else if (maxc == r) {
        *h = 60.0f * fmodf(((g - b) / delta), 6.0f);
    } else if (maxc == g) {
        *h = 60.0f * (((b - r) / delta) + 2.0f);
    } else {
        *h = 60.0f * (((r - g) / delta) + 4.0f);
    }
    if (*h < 0) *h += 360.0f;
}

/* Umbrales DEFINITIVOS. Calibrados con el sistema entero fijo desde el
 * boot: exposicion fija (6000us), foco fijo (886), WB fijo (1.603/1.400).
 * Sin clipping en ninguna tapita (max 163/255) y repetibilidad de +-0.3
 * grados sobre 3 repeticiones con reinicio de placa entre medio:
 *   rojo     H=5.1-5.2      S=0.76
 *   naranja  H=23.2-23.5    S=0.77-0.80
 *   amarillo H=60.4-60.6    S=0.97
 *   azul     H=227.0-227.3  S=0.77
 *   lila     H=306.5-308.7  S=0.50-0.55
 *   blanco   H=330-344      S=0.06-0.07
 *
 * El chequeo de saturacion va PRIMERO a proposito: con S casi nulo el
 * matiz no significa nada (el blanco llega a medir H=344, que por matiz
 * caeria en rojo). La saturacion separa blanco de lila sin ambiguedad:
 * 0.07 contra 0.50.
 */
static const char *classify_hsv(float h, float s, float v)
{
    if (v < 0.15f) return "muy oscuro / sin luz suficiente";

    if (s < 0.30f) {
        return (v > 0.35f) ? "blanco" : "gris / indefinido";
    }

    /* El matiz es circular (360 grados = 0 grados): un rojo real puede
     * medir H cercano a 360 en vez de cercano a 0 (ej. H=357.7 en una
     * tapita roja rota real, ver NOTAS_TINYML.md / historial del
     * proyecto). Sin este tramo final, cualquier H entre ~309 y 360 caia
     * en "lila" por descarte -- el margen 346-360 es simetrico al que ya
     * existia del otro lado del rojo (h<14), y deja un colchon de sobra
     * respecto del lila calibrado (306.5-308.7). */
    /* Corte rojo/naranja en 7 (antes 14): con la luz del aula las naranjas
     * bajan a H=9-13, y con la de casa las rojas llegan a H=5. */
    if (h < 7.0f)   return "rojo";
    if (h < 42.0f)  return "naranja";
    if (h < 150.0f) return "amarillo";
    if (h < 270.0f) return "azul";
    if (h < 346.0f) return "lila";
    return "rojo";
}

static void classify_and_log(const uint8_t *frame, uint32_t w, uint32_t h)
{
    float r, g, b, hh, s, v;
    sample_average_rgb_region(frame, w, SAMPLE_X0(w), SAMPLE_Y0(h), SAMPLE_WINDOW, SAMPLE_WINDOW, &r, &g, &b);
    rgb_to_hsv(r, g, b, &hh, &s, &v);
    const char *color = classify_hsv(hh, s, v);

    ESP_LOGI(TAG, "RGB=(%.2f,%.2f,%.2f) HSV: H=%.1f S=%.3f V=%.3f", r, g, b, hh, s, v);
    ESP_LOGI(TAG, "===== COLOR DETECTADO: %s =====", color);

    /* Guardado para MQTT: camera_capture() publica esto cuando send_image
     * es true, no en cada 'c' de prueba. */
    s_last_color = color;
    s_last_h = hh;
    s_last_s = s;
    s_last_v = v;

    /* El estado del WB va en cada medicion a proposito: una misma tapita da
     * numeros muy distintos con o sin la ganancia aplicada (visto en la
     * practica: la naranja pasaba de B=87 a B=27 y se confundia con amarillo).
     * Sin este dato al lado, un numero raro no se puede interpretar. */
    if (s_manual_wb_active) {
        ESP_LOGI(TAG, "  [wb] FIJO red=%.3f blue=%.3f", s_red_gain, s_blue_gain);
    } else {
        ESP_LOGW(TAG, "  [wb] AUTOMATICO -- los umbrales de color NO son validos asi. Apreta 'j'.");
    }

    /* Ayuda para calibrar: si estas apuntando a un blanco/gris neutro,
     * R, G y B deberian quedar casi iguales una vez que el WB este bien. */
    if (s_manual_wb_active) {
        ESP_LOGI(TAG, "  [calibracion] apuntando a blanco/gris: R-G=%.0f  B-G=%.0f "
        "(objetivo: ambos cerca de 0)", r - g, b - g);
    }
}

/* ---------- Envio de imagen por serial (para el comando 'i') ------------- */

static esp_err_t serial_send_jpeg(const char *name, const uint8_t *rgb565, uint32_t w, uint32_t h)
{
    jpeg_encoder_handle_t enc = NULL;
    jpeg_encode_engine_cfg_t eng = { .timeout_ms = 5000 };
    esp_err_t ret = jpeg_new_encoder_engine(&eng, &enc);
    ESP_RETURN_ON_ERROR(ret, TAG, "jpeg engine");

    size_t cap = w * h;
    jpeg_encode_memory_alloc_cfg_t mem = { .buffer_direction = JPEG_ENC_ALLOC_OUTPUT_BUFFER };
    size_t out_alloc = 0;
    uint8_t *out = jpeg_alloc_encoder_mem(cap, &mem, &out_alloc);
    if (!out) {
        jpeg_del_encoder_engine(enc);
        return ESP_ERR_NO_MEM;
    }

    jpeg_encode_cfg_t cfg = {
        .width = w, .height = h,
        .src_type = JPEG_ENCODE_IN_FORMAT_RGB565,
        .sub_sample = JPEG_DOWN_SAMPLING_YUV422,
        .image_quality = JPEG_QUALITY,
    };
    uint32_t out_len = 0;
    ret = jpeg_encoder_process(enc, &cfg, rgb565, w * h * 2, out, out_alloc, &out_len);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "JPEG q%d: %" PRIu32 " -> %" PRIu32 " bytes", JPEG_QUALITY, w * h * 2, out_len);
        ret = imx_serial_send_blob(name, "jpeg", w, h, out, out_len, NULL);

        /* La imagen NO se manda por MQTT/WiFi a proposito: transferencias de
         * ~90KB+ por el enlace SDIO P4<->C6 disparan un bug conocido y sin
         * resolver de esp-hosted-mcu ("Unrecoverable host sdio state",
         * issues #167/#184 del repo oficial) que reinicia la placa. La
         * imagen sigue llegando bien por serial, como siempre; por WiFi
         * viaja solo el JSON de clasificacion (unos 150 bytes, lejos de
         * la zona de riesgo). */
    } else {
        ESP_LOGE(TAG, "JPEG encode failed: %s", esp_err_to_name(ret));
    }

    free(out);
    jpeg_del_encoder_engine(enc);
    return ret;
}

/* ---------- Camara: init una sola vez, stream continuo ------------------- */

static esp_err_t camera_open_and_prepare(camera_ctx_t *ctx)
{
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (esp_video_init(&cam_config) != ESP_OK) {
        ESP_LOGE(TAG, "esp_video_init failed");
        return ESP_FAIL;
    }

    ctx->fd = open(CAM_DEV_PATH, O_RDONLY);
    if (ctx->fd < 0) {
        ESP_LOGE(TAG, "no se pudo abrir %s", CAM_DEV_PATH);
        return ESP_FAIL;
    }

    /* Nodo ISP separado: aca viven los controles de balance de blancos,
     * CCM, gamma, etc. No se usa para streaming, solo para ioctls. */
    s_isp_fd = open(ISP_DEV_PATH, O_RDWR);
    if (s_isp_fd < 0) {
        ESP_LOGW(TAG, "no se pudo abrir %s (errno=%d %s) -- el control de WB manual no va a funcionar",
                 ISP_DEV_PATH, errno, strerror(errno));
    } else {
        ESP_LOGI(TAG, "nodo ISP abierto en %s", ISP_DEV_PATH);
    }

    struct v4l2_format fmt = { .type = type };
    ioctl(ctx->fd, VIDIOC_G_FMT, &fmt);
    ctx->width = fmt.fmt.pix.width;
    ctx->height = fmt.fmt.pix.height;
    ESP_LOGI(TAG, "formato: %" PRIu32 "x%" PRIu32, ctx->width, ctx->height);

    struct v4l2_requestbuffers req = { .count = BUFFER_COUNT, .type = type, .memory = V4L2_MEMORY_MMAP };
    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) != 0) {
        ESP_LOGE(TAG, "VIDIOC_REQBUFS fallo");
        return ESP_FAIL;
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        struct v4l2_buffer b = { .type = type, .memory = V4L2_MEMORY_MMAP, .index = i };
        if (ioctl(ctx->fd, VIDIOC_QUERYBUF, &b) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QUERYBUF fallo");
            return ESP_FAIL;
        }
        ctx->buffer[i] = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, b.m.offset);
        if (!ctx->buffer[i]) {
            ESP_LOGE(TAG, "mmap fallo");
            return ESP_FAIL;
        }
        ctx->buffer_len[i] = b.length;
        /* Limpieza de cache antes de que el DMA empiece a escribir ahi
         * (ver comentario extenso del autor original sobre esto). */
        ESP_ERROR_CHECK(esp_cache_msync(ctx->buffer[i], (b.length / 64) * 64,
                                        ESP_CACHE_MSYNC_FLAG_DIR_C2M | ESP_CACHE_MSYNC_FLAG_INVALIDATE));
        ioctl(ctx->fd, VIDIOC_QBUF, &b);
    }

    if (ioctl(ctx->fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "VIDIOC_STREAMON fallo");
        return ESP_FAIL;
    }

    return ESP_OK;
}

static void camera_capture(camera_ctx_t *ctx, bool send_image)
{
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    struct v4l2_buffer buf;

    if (s_camera_mutex) {
        xSemaphoreTake(s_camera_mutex, portMAX_DELAY);
    }

    /* Diagnostico: leer la posicion del VCM antes de descartar frames y
     * de nuevo justo antes de clasificar. Si difieren, el autofoco seguia
     * moviendose en el momento de la captura -- eso explicaria una foto
     * desenfocada sin que sea un problema de exposicion/movimiento. */
    struct v4l2_ext_control focus_ctrl = { .id = V4L2_CID_FOCUS_ABSOLUTE };
    struct v4l2_ext_controls focus_ctrls = {
        .ctrl_class = V4L2_CID_CAMERA_CLASS, .count = 1, .controls = &focus_ctrl,
    };
    int focus_before = (ioctl(ctx->fd, VIDIOC_G_EXT_CTRLS, &focus_ctrls) == 0) ? focus_ctrl.value : -1;

    for (int i = 0; i < DISCARD_FRAMES; i++) {
        buf = (struct v4l2_buffer){ .type = type, .memory = V4L2_MEMORY_MMAP };
        if (ioctl(ctx->fd, VIDIOC_DQBUF, &buf) != 0) {
            ESP_LOGE(TAG, "DQBUF fallo");
            if (s_camera_mutex) {
                xSemaphoreGive(s_camera_mutex);
            }
            return;
        }

        /* CRITICO: el DMA acaba de escribir el frame nuevo en RAM, pero la
         * CPU tiene datos viejos en su cache. Sin esta invalidacion se lee
         * el frame anterior -- sintoma observado: el mismo pixel exacto,
         * bit por bit, capture tras capture, sin importar la exposicion. */
        esp_cache_msync(ctx->buffer[buf.index],
                        (ctx->buffer_len[buf.index] / 64) * 64,
                        ESP_CACHE_MSYNC_FLAG_DIR_M2C);

        if (i == DISCARD_FRAMES - 1) {
            int focus_after = (ioctl(ctx->fd, VIDIOC_G_EXT_CTRLS, &focus_ctrls) == 0) ? focus_ctrl.value : -1;
            ESP_LOGI(TAG, "foco: %d -> %d %s", focus_before, focus_after,
                     (focus_before != focus_after) ? "(SE ESTABA MOVIENDO)" : "(estable)");

            const uint8_t *frame = ctx->buffer[buf.index];
            classify_and_log(frame, ctx->width, ctx->height);
            if (send_image) {
                /* TinyML sana/rota: mismo criterio que el color, solo en
                 * capturas "reales" -- correrlo en cada 'c' de prueba
                 * gastaria ciclos sin necesidad. */
                float prob_rota = -1.0f;
                if (tinyml_classify_estado(frame, ctx->width, ctx->height, &prob_rota)) {
                    s_last_prob_rota = prob_rota;
                    /* UMBRAL_ROTA_LOCAL/CONFIANZA_MINIMA_LOCAL son solo para
                     * que el log diga algo legible mientras se prueba en el
                     * banco -- el valor que de verdad decide sana/rota/dudoso
                     * es el que se aplique del lado de la Raspberry
                     * (tapitas_ingest.py) sobre el prob_rota crudo que se
                     * manda por MQTT, asi se puede ajustar sin reflashear el
                     * ESP32. Ver threshold_results.txt: 0.20 recomendado. */
                    const float UMBRAL_ROTA_LOCAL = 0.20f;
                    const float CONFIANZA_MINIMA_LOCAL = 0.50f;
                    float confianza_local = (prob_rota > UMBRAL_ROTA_LOCAL)
                                             ? prob_rota : (1.0f - prob_rota);
                    const char *estado_local =
                        (confianza_local < CONFIANZA_MINIMA_LOCAL) ? "DUDOSO" :
                        (prob_rota > UMBRAL_ROTA_LOCAL) ? "ROTA" : "SANA";
                    ESP_LOGI(TAG, "  [tinyml] prob_rota=%.4f  -> ESTADO: %s", prob_rota,
                             estado_local);
                } else {
                    ESP_LOGW(TAG, "  [tinyml] clasificacion fallo (modelo no inicializado?)");
                    s_last_prob_rota = -1.0f;
                }

                serial_send_jpeg("imx708", frame, ctx->width, ctx->height);
                /* Publicar por MQTT solo en capturas "reales" (send_image=true:
                 * disparo automatico o 'i'), no en cada 'c' de prueba manual. */
                publish_clasificacion_por_serial();
            }
        }

        ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    }

    if (s_camera_mutex) {
        xSemaphoreGive(s_camera_mutex);
    }
}

/* Tarea aparte: mientras s_streaming este activo, dispara capturas+envio
 * de imagen en loop, sin que el usuario tenga que apretar 'i' cada vez.
 * El propio tiempo de "camera_capture" (JPEG + envio serial ~1-1.5s)
 * ya marca el ritmo, no hace falta un delay extra ahi. */
static void streaming_task(void *arg)
{
    camera_ctx_t *ctx = (camera_ctx_t *)arg;
    while (1) {
        if (s_streaming) {
            camera_capture(ctx, true);
        } else {
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }
}

void app_main(void)
{
    camera_ctx_t ctx = {0};
    ESP_ERROR_CHECK(camera_open_and_prepare(&ctx));

    s_cam_ctx = &ctx;
    s_camera_mutex = xSemaphoreCreateMutex();

    /* TinyML sana/rota: reserva el tensor arena e inicializa el interprete
     * UNA sola vez, aca. Si falla (memoria, modelo corrupto) queda logueado
     * y las capturas siguen andando solo con el color HSV -- se degrada,
     * no se cae todo el sistema. */
    tinyml_init();

    /* Guardar la config AWB de fabrica ANTES de tocarla -- la necesitamos
     * intacta para poder reactivar el AWB mas adelante con 'n'. */
    isp_save_default_awb_config(s_isp_fd);

    /* Centrar la ventana de estadisticas AE+AF sobre donde cae la tapita.
     * Sin esto, el AE promedia el fondo oscuro y sobreexpone la tapita. */
    set_center_stats_window(ctx.width, ctx.height);

    /* Exposicion fija desde el arranque: mismo valor siempre, sin depender
     * de que el AGC converja ni de apretar teclas. Ajustable con '1'/'2'. */
    set_fixed_exposure(s_fixed_exposure_us);

    /* Foco fijo: se apaga el autofoco y se clava en el valor calibrado.
     * Sin esto tardaba ~8s en enfocar cada vez que cambiaba la tapita. */
    disable_autofocus(ctx.fd, s_isp_fd);
    set_manual_focus(ctx.fd, CALIBRATED_FOCUS);

    /* Balance de blancos fijo, sin apagar las estadisticas (apagarlas
     * frenaba tambien la auto-exposicion). */
    s_red_gain = CALIBRATED_RED_GAIN;
    s_blue_gain = CALIBRATED_BLUE_GAIN;
    isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
    s_manual_wb_active = true;

    ESP_LOGI(TAG, "===== SISTEMA LISTO SIN INTERVENCION =====");
    ESP_LOGI(TAG, "  exposicion=%" PRIu32 "us (fija)  foco=%d (fijo)  wb=%.3f/%.3f (fijo)",
             s_fixed_exposure_us, CALIBRATED_FOCUS, s_red_gain, s_blue_gain);

    /* WiFi/MQTT sacados del ESP32 por completo -- eran la causa confirmada
     * de los reinicios sin fondo por un bug de esp-hosted-mcu (P4+C6 SDIO)
     * sin resolver. La clasificacion ahora viaja por el cable serial de
     * siempre (ver publish_clasificacion_por_serial), y ver_imx708.py en
     * la PC hace de puente hacia la Raspberry por MQTT. */
    snprintf(s_sesion_id, sizeof(s_sesion_id), "ses-%08" PRIx32, esp_random());
    ESP_LOGI(TAG, "sesion_id=%s", s_sesion_id);

    /* Se arranca en AWB AUTOMATICO a proposito. Aplicar una ganancia fija
     * al boot dejaba la imagen clavada en el estado "amarillo" sin salida.
     * El flujo correcto: dejar que el AWB corra, hacer el ritual de la luz
     * hasta que la imagen se vea bien, y ahi congelar con 'm'.
     * 'f' aplica las ganancias calibradas de una sesion anterior, si servian. */

    /* &ctx vive en el stack de app_main, que nunca retorna (el while(1) de
     * abajo corre para siempre), asi que es seguro que streaming_task la
     * siga usando indefinidamente. */
    xTaskCreate(streaming_task, "streaming_task", 4096, &ctx, 5, NULL);

    ir_sensor_init();
    motor_init();
    xTaskCreate(belt_task, "belt_task", 2048, NULL, 5, NULL);
    xTaskCreate(capture_trigger_task, "capture_trigger_task", 4096, NULL, 6, NULL);

    ESP_LOGI(TAG, "Listo.");
    ESP_LOGI(TAG, "  'c' = clasificar rapido   'i' = clasificar + ver imagen");
    ESP_LOGI(TAG, "  'v' = prender/apagar streaming en loop (~1 foto cada 1-2s)");
    ESP_LOGI(TAG, "  'l' = listar controles del nodo ISP   'L' = listar controles del sensor/camara");
    ESP_LOGI(TAG, "  'm' = activar balance de blancos MANUAL (apaga el AWB automatico)");
    ESP_LOGI(TAG, "  'r'/'R' = red_gain -/+   'b'/'B' = blue_gain -/+   'p' = ver valores");
    ESP_LOGI(TAG, "  't' = probar motor ADELANTE (%d pasos)   'T' = probar motor ATRAS", MOTOR_TEST_STEPS);
    ESP_LOGI(TAG, "  'g' = arrancar cinta continua   's' = parar cinta");
    ESP_LOGI(TAG, "  'a' = prender/apagar disparo automatico (sensor -> captura sola)");
    ESP_LOGI(TAG, "  'y'/'Y' = bajar/subir el delay de disparo (paso %dms)", TRIGGER_DELAY_STEP_MS);
    ESP_LOGI(TAG, "  'k'/'K' = bajar/subir el tiempo de asentado tras frenar (paso %dms)", SETTLE_DELAY_STEP_MS);
    ESP_LOGI(TAG, "  'n' = volver a AWB automatico (para rehacer el ritual de la luz)");
    ESP_LOGI(TAG, "  'm' = congelar el color actual   'f' = aplicar ganancias calibradas guardadas");
    ESP_LOGI(TAG, "  'x' = ver limites de exposicion   'w'/'W' = bajar/subir el techo de exposicion");
    ESP_LOGI(TAG, "  'q'/'Q' = bajar/subir el PISO de exposicion (este es el que muerde si satura)");
    ESP_LOGI(TAG, "  'j' = WB fijo SIN apagar stats (auto-exposicion sigue viva) -- PROBAR ESTE");
    ESP_LOGI(TAG, "  'z' = recentrar ventana de estadisticas AE+AF (ya se aplica sola al boot)");
    ESP_LOGI(TAG, "  'u' = WB neutro 1.0/1.0 (diagnostico: ver si el clipping lo causan las ganancias)");
    ESP_LOGI(TAG, "  'o'/'O' = foco manual -/+ (paso %d)", FOCUS_STEP);
    ESP_LOGI(TAG, "  'd' = CONGELAR exposicion   'D' = descongelar (volver a automatica)");
    ESP_LOGI(TAG, "  '1'/'2' = bajar/subir la EXPOSICION FIJA (actual %" PRIu32 "us) -- usar estos", s_fixed_exposure_us);

    while (1) {
        int ch = fgetc(stdin);
        if (ch == 'c') {
            if (s_streaming) {
                ESP_LOGW(TAG, "Apaga el streaming con 'v' antes de capturar manual.");
            } else {
                ESP_LOGI(TAG, "Capturando...");
                camera_capture(&ctx, false);
            }
        } else if (ch == 'i') {
            if (s_streaming) {
                ESP_LOGW(TAG, "Apaga el streaming con 'v' antes de capturar manual.");
            } else {
                ESP_LOGI(TAG, "Capturando + imagen...");
                camera_capture(&ctx, true);
            }
        } else if (ch == 'v' || ch == 'V') {
            s_streaming = !s_streaming;
            if (s_streaming) {
                ESP_LOGI(TAG, "===== STREAMING ON ===== (foto cada 1-2s aprox, limitado por el serial. "
                "'v' de nuevo para parar. Podes seguir usando 'r'/'R'/'b'/'B' mientras tanto.)");
            } else {
                ESP_LOGI(TAG, "===== STREAMING OFF =====");
            }
        } else if (ch == 'l') {
            list_available_controls(s_isp_fd, ISP_DEV_PATH);
        } else if (ch == 'L') {
            list_available_controls(s_cam_ctx->fd, CAM_DEV_PATH);
        } else if (ch == 'm' || ch == 'M') {
            enable_manual_white_balance();
        } else if (ch == 'r') {
            adjust_red_gain(-WB_GAIN_STEP);
        } else if (ch == 'R') {
            adjust_red_gain(+WB_GAIN_STEP);
        } else if (ch == 'b') {
            adjust_blue_gain(-WB_GAIN_STEP);
        } else if (ch == 'B') {
            adjust_blue_gain(+WB_GAIN_STEP);
        } else if (ch == 'p' || ch == 'P') {
            print_current_wb();
        } else if (ch == 't') {
            if (s_belt_running) {
                ESP_LOGW(TAG, "Para la cinta continua ('s') antes de probar pasos manuales.");
            } else {
                ESP_LOGI(TAG, "Motor: probando %d pasos adelante...", MOTOR_TEST_STEPS);
                motor_step(MOTOR_TEST_STEPS, true, MOTOR_TEST_STEP_DELAY_US);
            }
        } else if (ch == 'T') {
            if (s_belt_running) {
                ESP_LOGW(TAG, "Para la cinta continua ('s') antes de probar pasos manuales.");
            } else {
                ESP_LOGI(TAG, "Motor: probando %d pasos atras...", MOTOR_TEST_STEPS);
                motor_step(MOTOR_TEST_STEPS, false, MOTOR_TEST_STEP_DELAY_US);
            }
        } else if (ch == 'g') {
            gpio_set_level(MOTOR_DIR_GPIO, 0);  /* adelante, con la inversion ya aplicada */
            s_belt_running = true;
            ESP_LOGI(TAG, "===== CINTA: arrancando continua =====");
        } else if (ch == 's') {
            s_belt_running = false;
            ESP_LOGI(TAG, "===== CINTA: parada =====");
        } else if (ch == 'a' || ch == 'A') {
            s_auto_capture_enabled = !s_auto_capture_enabled;
            ESP_LOGI(TAG, "===== DISPARO AUTOMATICO: %s (delay actual %" PRIu32 "ms) =====",
                     s_auto_capture_enabled ? "ON" : "OFF", s_trigger_delay_ms);
        } else if (ch == 'y') {
            if (s_trigger_delay_ms >= TRIGGER_DELAY_STEP_MS) {
                s_trigger_delay_ms -= TRIGGER_DELAY_STEP_MS;
            }
            ESP_LOGI(TAG, "Delay de disparo: %" PRIu32 "ms", s_trigger_delay_ms);
        } else if (ch == 'Y') {
            s_trigger_delay_ms += TRIGGER_DELAY_STEP_MS;
            ESP_LOGI(TAG, "Delay de disparo: %" PRIu32 "ms", s_trigger_delay_ms);
        } else if (ch == 'k') {
            if (s_settle_delay_ms >= SETTLE_DELAY_STEP_MS) {
                s_settle_delay_ms -= SETTLE_DELAY_STEP_MS;
            }
            ESP_LOGI(TAG, "Tiempo de asentado: %" PRIu32 "ms", s_settle_delay_ms);
        } else if (ch == 'K') {
            s_settle_delay_ms += SETTLE_DELAY_STEP_MS;
            ESP_LOGI(TAG, "Tiempo de asentado: %" PRIu32 "ms", s_settle_delay_ms);
        } else if (ch == 'n' || ch == 'N') {
            isp_enable_awb_stats(s_isp_fd);
        } else if (ch == 'f' || ch == 'F') {
            s_red_gain = CALIBRATED_RED_GAIN;
            s_blue_gain = CALIBRATED_BLUE_GAIN;
            isp_disable_awb_stats(s_isp_fd);
            isp_apply_manual_wb(s_isp_fd, s_red_gain, s_blue_gain);
            s_manual_wb_active = true;
            ESP_LOGI(TAG, "Ganancias calibradas guardadas aplicadas.");
        } else if (ch == 'j' || ch == 'J') {
            apply_wb_keeping_stats();
        } else if (ch == 'z' || ch == 'Z') {
            set_center_stats_window(s_cam_ctx->width, s_cam_ctx->height);
        } else if (ch == 'o') {
            adjust_manual_focus(s_cam_ctx->fd, -FOCUS_STEP);
        } else if (ch == 'O') {
            adjust_manual_focus(s_cam_ctx->fd, +FOCUS_STEP);
        } else if (ch == 'u' || ch == 'U') {
            apply_neutral_wb();
        } else if (ch == 'x' || ch == 'X') {
            probe_exposure_controls(s_cam_ctx->fd);
            show_exposure_limits();
            show_agc_status();
        } else if (ch == '1') {
            set_fixed_exposure((uint32_t)(s_fixed_exposure_us * 0.8f));
        } else if (ch == '2') {
            set_fixed_exposure((uint32_t)(s_fixed_exposure_us * 1.25f));
        } else if (ch == 'd') {
            freeze_exposure();
        } else if (ch == 'D') {
            unfreeze_exposure();
        } else if (ch == 'q') {
            adjust_exposure_floor(0.5f);   /* bajar el piso a la mitad */
        } else if (ch == 'Q') {
            adjust_exposure_floor(2.0f);   /* subir el piso al doble */
        } else if (ch == 'w') {
            adjust_max_exposure(EXPOSURE_STEP_FACTOR);        /* bajar techo */
        } else if (ch == 'W') {
            adjust_max_exposure(1.0f / EXPOSURE_STEP_FACTOR); /* subir techo */
        } else if (ch == 'e') {
            adjust_exposure(s_cam_ctx->fd, 0.75f);   /* bajar exposicion (via V4L2, si estuviera) */
        } else if (ch == 'E') {
            adjust_exposure(s_cam_ctx->fd, 1.33f);   /* subir exposicion (via V4L2, si estuviera) */
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
