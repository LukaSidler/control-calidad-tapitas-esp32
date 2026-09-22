idf_component_register(
    SRCS "imx708_snapshot_main.c" "tinyml_estado.cc" "tapita_model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES esp_video esp_ipa esp_cam_sensor_imx imx_serial_img fatfs esp_driver_sdmmc esp_driver_jpeg esp_driver_uart
    PRIV_REQUIRES esp-tflite-micro esp_timer
)

# tinyml_estado.cc usa la API C++ de TFLite Micro (namespaces, clases) --
# necesita C++17. Sin esto, en algunas versiones de sdkconfig el default
# del proyecto puede quedar en un estandar mas viejo y tirar errores de
# compilacion raros en los headers de tensorflow/lite/micro/.
set_property(TARGET ${COMPONENT_LIB} PROPERTY CXX_STANDARD 17)

# Fail the build if the IMX708 ISP tuning config did not resolve. esp_ipa's own
# existence check is spelled message(FETAL_ERROR ...) - not a real CMake mode -
# so it prints and carries on, and a missing file shows up much later as a
# stream with no auto-exposure and no white balance rather than as an error.
# By the time this component is configured the path is final, so check it here.
idf_build_get_property(imx708_ipa_json IMX708_IPA_JSON_RESOLVED)
idf_build_get_property(imx708_project_dir PROJECT_DIR)
get_filename_component(imx708_ipa_abs "${imx708_ipa_json}"
    ABSOLUTE BASE_DIR "${imx708_project_dir}")
if(NOT EXISTS "${imx708_ipa_abs}")
    message(FATAL_ERROR
        "IMX708 ISP tuning config not found at:\n  ${imx708_ipa_abs}\n"
        "esp_ipa would silently skip it and the camera would run with no "
        "auto-exposure and no white balance. Check the ESP_IPA_JSON_CONFIG_FILE_PATH "
        "block in the project CMakeLists.txt.")
endif()
