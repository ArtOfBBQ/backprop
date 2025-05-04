#include <stdio.h>
#include <stdlib.h>

#include "toknnet.h"

int main(void) {
    
    toknn_init(malloc);
    
    TOKNN_ACTIVATION_FUNCTION activations[3];
    activations[0] = TOKNN_ACTIVATION_TANH;
    activations[1] = TOKNN_ACTIVATION_TANH;
    activations[2] = TOKNN_ACTIVATION_NONE;
    
    uint32_t nodes_per_layer[3];
    nodes_per_layer[0] = 8;
    nodes_per_layer[1] = 5;
    nodes_per_layer[2] = 1;
    uint32_t features_size = 2;
    
    toknn_create_model(
        /* const TOKNN_LOSS_FUNCTION loss_function: */
            TOKNN_LOSS_SQUAREDERROR,
        /* const TOKNN_ACTIVATION_FUNCTION * activation_functions: */
            activations,
        /* const uint32_t layers_size: */
            3,
        /* const uint32_t * nodes_per_layer: */
            nodes_per_layer,
        /* const uint32_t features_size: */
            features_size);
    
    float new_observation[features_size];
    new_observation[0] = 10.0f;
    new_observation[1] = 10.0f;
    
    float labels[200] = {
        -10.0f,-10.0f,-20.0f,-10.0f,-12.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,
        -10.0f,-10.0f,-10.0f,-10.2f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,
        -10.0f,-10.1f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.1f,
        -10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,-10.0f,
        10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,
        10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.1f,
        10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,
        10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,10.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,5.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.5f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.2f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.5f,0.0f,0.1f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.1f,0.0f,0.3f,0.0f,0.0f,0.0f,0.0f,0.0f,0.1f,0.0f,
        0.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.2f,0.0f,0.0f,0.0f,0.0f,0.0f,0.2f
        };
    
    float train[400] = {
        10.0f,-8.2f,5.0f,-7.1f,-5.0f,-2.3f,0.0f,-4.8f,8.0f,-10.1f,
        7.0f,-9.3f,3.0f,-6.2f,-3.0f,-3.1f,2.0f,-5.4f,9.0f,-8.7f,
        6.0f,-7.8f,4.0f,-6.9f,-4.0f,-2.7f,1.0f,-4.3f,10.0f,-9.5f,
        8.0f,-8.9f,5.0f,-7.4f,-2.0f,-3.8f,0.0f,-5.1f,7.0f,-9.2f,
        9.0f,-8.6f,6.0f,-7.7f,-1.0f,-4.2f,3.0f,-6.1f,8.0f,-9.0f,
        4.0f,-7.3f,2.0f,-5.9f,-5.0f,-3.0f,1.0f,-4.7f,9.0f,-8.8f,
        7.0f,-9.1f,5.0f,-7.6f,-3.0f,-3.5f,0.0f,-5.2f,6.0f,-8.4f,
        8.0f,-9.4f,4.0f,-7.0f,-2.0f,-4.1f,1.0f,-5.3f,9.0f,-8.9f,
        10.0f,2.3f,5.0f,4.2f,-5.0f,9.1f,0.0f,5.4f,8.0f,1.7f,
        7.0f,3.1f,3.0f,5.2f,-3.0f,8.3f,2.0f,4.8f,9.0f,2.6f,
        6.0f,3.9f,4.0f,5.0f,-4.0f,7.7f,1.0f,4.3f,10.0f,2.8f,
        8.0f,3.4f,5.0f,4.9f,-2.0f,6.8f,0.0f,5.1f,7.0f,3.2f,
        9.0f,2.9f,6.0f,4.1f,-1.0f,6.2f,3.0f,5.3f,8.0f,3.0f,
        4.0f,4.7f,2.0f,5.9f,-5.0f,7.0f,1.0f,4.5f,9.0f,2.7f,
        7.0f,3.5f,5.0f,4.6f,-3.0f,6.9f,0.0f,5.0f,6.0f,3.8f,
        8.0f,2.4f,4.0f,5.5f,-2.0f,7.1f,1.0f,4.4f,9.0f,3.3f,
        0.0f,0.2f,2.0f,-0.8f,-2.0f,0.9f,5.0f,-2.1f,-5.0f,2.4f,
        1.0f,-0.3f,-1.0f,0.4f,3.0f,-1.2f,-3.0f,1.1f,4.0f,-1.8f,
        6.0f,-0.7f,-4.0f,1.9f,0.0f,0.5f,2.0f,-0.9f,-2.0f,1.0f,
        5.0f,-2.0f,-5.0f,2.2f,1.0f,-0.6f,-1.0f,0.7f,3.0f,-1.3f,
        -3.0f,1.4f,4.0f,-1.7f,6.0f,-0.8f,-4.0f,2.0f,0.0f,0.3f,
        2.0f,-1.0f,-2.0f,0.8f,5.0f,-2.3f,-5.0f,2.5f,1.0f,-0.4f,
        -1.0f,0.6f,3.0f,-1.4f,-3.0f,1.5f,4.0f,-1.6f,6.0f,-0.9f,
        -4.0f,1.8f,0.0f,0.1f,2.0f,-1.1f,-2.0f,1.2f,5.0f,-2.2f,
        -5.0f,2.3f,1.0f,-0.5f,-1.0f,0.8f,3.0f,-1.5f,-3.0f,1.6f,
        4.0f,-1.9f,6.0f,-0.6f,-4.0f,1.7f,0.0f,0.4f,2.0f,-0.7f,
        -2.0f,1.3f,5.0f,-2.4f,-5.0f,2.1f,1.0f,-0.2f,-1.0f,0.9f,
        3.0f,-1.6f,-3.0f,1.7f,4.0f,-1.5f,6.0f,-0.5f,-4.0f,1.6f,
        0.0f,0.6f,2.0f,-0.5f,-2.0f,1.4f,5.0f,-2.5f,-5.0f,2.0f,
        1.0f,-0.1f,-1.0f,1.0f,3.0f,-1.7f,-3.0f,1.8f,4.0f,-1.4f,
        6.0f,-0.4f,-4.0f,1.5f,0.0f,0.7f,2.0f,-0.6f,-2.0f,1.5f,
        5.0f,-2.6f,-5.0f,1.9f,1.0f,-0.0f,-1.0f,1.1f,3.0f,-1.8f,
        -3.0f,1.9f,4.0f,-1.3f,6.0f,-0.3f,-4.0f,1.4f,0.0f,0.8f,
        2.0f,-0.4f,-2.0f,1.6f,5.0f,-2.7f,-5.0f,1.8f,1.0f,0.1f,
        -1.0f,1.2f,3.0f,-1.9f,-3.0f,2.0f,4.0f,-1.2f,6.0f,-0.2f,
        -4.0f,1.3f,0.0f,0.9f,2.0f,-0.3f,-2.0f,1.7f,5.0f,-2.8f,
        -5.0f,1.7f,1.0f,0.2f,-1.0f,1.3f,3.0f,-2.0f,-3.0f,2.1f,
        4.0f,-1.1f,6.0f,-0.1f,-4.0f,1.2f,0.0f,1.0f,2.0f,-0.2f,
        -2.0f,1.8f,5.0f,-2.9f,-5.0f,1.6f,1.0f,0.3f,-1.0f,1.4f,
        3.0f,-2.1f,-3.0f,2.2f,4.0f,-1.0f,6.0f,0.0f,-4.0f,1.1f
    };
    
    toknn_train(
        /* const int16_t * train: */
            train,
        /* const uint32_t train_size: */
            400,
        /* const int16_t * labels: */
            labels,
        /* const uint32_t labels_size: */
            200);
    
    return 0;
}

