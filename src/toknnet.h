#ifndef TOKNN_H
#define TOKNN_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/*
Call this function to initialize and allocate memory
*/
void toknn_init(
    void *(* arg_toknn_malloc_func)(size_t));

typedef enum TOKNN_LOSS_FUNCTION {
    TOKNN_LOSS_UNDERFLOW,
    TOKNN_LOSS_ABSOLUTEERROR,
    TOKNN_LOSS_SQUAREDERROR,
    TOKNN_LOSS_OVERFLOW,
} TOKNN_LOSS_FUNCTION;

typedef enum TOKNN_ACTIVATION_FUNCTION {
    TOKNN_ACTIVATION_UNDERFLOW,
    TOKNN_ACTIVATION_NONE,
    TOKNN_ACTIVATION_TANH,
    TOKNN_ACTIVATION_DOUBLECLAMP_10000,
    TOKNN_ACTIVATION_DOUBLECLAMP_5000,
    TOKNN_ACTIVATION_DOUBLECLAMP_2000,
    TOKNN_ACTIVATION_DOUBLECLAMP_1000,
    TOKNN_ACTIVATION_DOUBLECLAMP_200,
    TOKNN_ACTIVATION_OVERFLOW,
} TOKNN_ACTIVATION_FUNCTION;

void toknn_create_model(
    const TOKNN_LOSS_FUNCTION loss_function,
    const TOKNN_ACTIVATION_FUNCTION * activation_functions,
    const uint32_t layers_size,
    const uint32_t * nodes_per_layer,
    const uint32_t features_size);

void toknn_train(
    const float * train,
    const uint32_t train_size,
    const float * labels,
    const uint32_t labels_size);

float toknn_predict(
    const float * new_observations,
    const uint32_t new_observations_size);

#endif // TOKNN_H

