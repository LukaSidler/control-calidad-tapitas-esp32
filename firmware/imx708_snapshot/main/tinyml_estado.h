/* Clasificacion sana/rota via TinyML (TFLite Micro), sobre el mismo frame
 * RGB565 que ya usa classify_and_log() para el color. Wrapper en C para
 * poder llamarse desde imx708_snapshot_main.c (que es .c, no .cc) -- la
 * implementacion real vive en tinyml_estado.cc y usa la API C++ de
 * TFLite Micro por debajo. */
#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reserva el tensor arena (PSRAM) y arma el interprete TFLite Micro.
 * Llamar UNA sola vez, en app_main(), antes de la primera captura.
 * Si falla, queda logueado por ESP_LOGE y tinyml_classify_estado()
 * despues siempre devuelve false -- el sistema sigue andando solo con
 * clasificacion de color. */
void tinyml_init(void);

/* Corre el modelo sana/rota sobre el frame RGB565 completo (frame_w x
 * frame_h, tal cual llega de la camara -- normalmente 1920x1080).
 *
 * Recorta 1120x1120 centrado en el medio geometrico del frame (mismo lugar
 * donde ya estan centrados SAMPLE_WINDOW y el STATS_WINDOW de AE/AF -- ahi
 * es donde cae la tapita siempre, con el timing de motor/sensor IR ya
 * calibrado), reduce a escala de grises 128x128 con el mismo criterio de
 * promediado que uso el entrenamiento, cuantiza a int8 con la escala del
 * propio modelo, y corre la inferencia.
 *
 * Devuelve true si pudo clasificar (false si tinyml_init() fallo o el
 * modelo no esta listo). prob_rota_out queda con la probabilidad de
 * "rota" (0.0 a 1.0) -- el umbral de decision NO se aplica aca a
 * proposito: se aplica rio abajo, en tapitas_ingest.py en la Raspberry,
 * para poder ajustarlo sin reflashear el ESP32. */
bool tinyml_classify_estado(const uint8_t *frame_rgb565, uint32_t frame_w,
                             uint32_t frame_h, float *prob_rota_out);

#ifdef __cplusplus
}
#endif
