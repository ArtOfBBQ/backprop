#include "toknnet.h"

static float loss_squarederror(
    const float y,
    const float t)
{
    return (y-t)*(y-t);
}

static float derivative_squarederror(
    const float y,
    const float t)
{
    /*
    L(y) = (y-t)^2
    dL/dy = d(y-t)/dy * 2*(y-t)
    dL/dy = 1 * 2 * (y-t)
    dL/dy = 2 * (y-t)
    */
    return 2.0f * (y-t);
}

static float loss_absoluteerror(
    const float y,
    const float t)
{
    float diff = y - t;
    return diff >= 0.0f ? diff : -diff;
}

static float derivative_absoluteerror(
    const float y,
    const float t)
{
    if (y > t) return  1.0f;
    if (y < t) return -1.0f;
    
    return 0.0f;
}

static float activation_doublecap_10000(
    const float in)
{
    if (in < -10000) { return -10000; }
    if (in >  10000) { return  10000; }
    return in;
}

static float derivative_doublecap_10000(
    const float x)
{
    if (x >  10000) { return 0; }
    if (x < -10000) { return 0; }
    
    return 1;
}

static float activation_doublecap_5000(
    const float in)
{
    if (in < -5000) { return -5000; }
    if (in >  5000) { return  5000; }
    return in;
}

static float derivative_doublecap_5000(
    const float x)
{
    if (x >  5000) { return 0; }
    if (x < -5000) { return 0; }
    
    return 1;
}

static float activation_doublecap_2000(
    const float in)
{
    if (in < -2000) { return -2000; }
    if (in >  2000) { return  2000; }
    return in;
}

static float derivative_doublecap_2000(
    const float x)
{
    if (x >  2000) { return 0; }
    if (x < -2000) { return 0; }
    
    return 1;
}

static float activation_doublecap_1000(
    const float in)
{
    if (in < -1000) { return -1000; }
    if (in >  1000) { return  1000; }
    return in;
}

static float derivative_doublecap_1000(
    const float x)
{
    if (x >  1000) { return 0; }
    if (x < -1000) { return 0; }
    
    return 1;
}

static float activation_doublecap_200(
    const float in)
{
    if (in < -200) { return -200; }
    if (in >  200) { return  200; }
    return in;
}

static float derivative_doublecap_200(
    const float x)
{
    if (x >  200) { return 0; }
    if (x < -200) { return 0; }
    
    return 1;
}

static float activation_tanh(const float in) {
    return tanhf(in);
}

static float derivative_tanh(const float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

#define TOKNN_MAX_LAYERS 3
#define TOKNN_MAX_NODES 500
#define TOKNN_MAX_WEIGHTS 1000
typedef struct TOKNeuralNet {
    TOKNN_LOSS_FUNCTION loss_function;
    TOKNN_ACTIVATION_FUNCTION activation_functions[TOKNN_MAX_LAYERS];
    uint32_t layers_size;
    uint32_t features_size;
    uint32_t nodes_size;
    uint32_t weights_size;
    uint32_t nodes_per_layer[TOKNN_MAX_LAYERS];
    float preactivated_node_values[TOKNN_MAX_NODES];
    float activated_node_values[TOKNN_MAX_NODES];
    float biases[TOKNN_MAX_NODES];
    float weights[TOKNN_MAX_WEIGHTS];
} TOKNeuralNet;

typedef struct OptimizerData {
    // For all observations in the training data
    float all_obs_node_and_bias_derivatives[TOKNN_MAX_NODES];
    float all_obs_weight_derivatives[TOKNN_MAX_WEIGHTS];
    // For the current observation only
    float this_obs_node_and_bias_derivatives[TOKNN_MAX_NODES];
    float this_obs_weight_derivatives[TOKNN_MAX_WEIGHTS];
} OptimizerData;

typedef struct LocalCache {
    TOKNeuralNet nn;
    TOKNeuralNet nn_debug;
    OptimizerData optim;
} LocalCache;

static LocalCache * cache = NULL;

void toknn_init(
    void *(* arg_toknn_malloc_func)(size_t))
{
    cache = arg_toknn_malloc_func(sizeof(LocalCache));
}

static uint32_t to_overall_node_i(
    const uint32_t in_layer_i,
    const uint32_t in_layer_node_i)
{
    assert(in_layer_node_i < TOKNN_MAX_NODES);
    
    TOKNeuralNet * nn = &cache->nn;
    
    uint32_t return_value = 0;
    
    // Add all nodes in the full layers
    for (
        uint32_t layer_i = 0;
        layer_i < TOKNN_MAX_LAYERS;
        layer_i++)
    {
        if (layer_i >= in_layer_i) { break; }
        assert(nn->nodes_per_layer[layer_i] < TOKNN_MAX_NODES);
        return_value += nn->nodes_per_layer[layer_i];
    }
    
    // Add partial nodes for the layer we're in
    return_value += in_layer_node_i;
    
    assert(return_value < TOKNN_MAX_NODES);
    return return_value;
}

static uint32_t to_overall_weight_i(
    const uint32_t in_layer_i,
    const uint32_t in_layer_node_i)
{
    TOKNeuralNet * nn = &cache->nn;
    
    assert(in_layer_i < nn->layers_size);
    assert(in_layer_node_i < nn->nodes_per_layer[in_layer_i]);
    
    uint32_t return_value = 0;
     
    // Add all nodes in the full layers
    for (
        uint32_t layer_i = 0;
        layer_i < TOKNN_MAX_LAYERS;
        layer_i++)
    {
        if (layer_i >= in_layer_i) { break; }
        
        if (layer_i == 0) {
            return_value += nn->nodes_per_layer[layer_i] *
                nn->features_size;
        } else {
            return_value +=  nn->nodes_per_layer[layer_i] *
                nn->nodes_per_layer[layer_i-1];
        }
    }
    
    // Add partial nodes for the layer we're in
    return_value += in_layer_i == 0 ?
        nn->features_size * in_layer_node_i :
        nn->nodes_per_layer[in_layer_i-1] * in_layer_node_i;
    
    assert(return_value < TOKNN_MAX_WEIGHTS);
    return return_value;
}

void toknn_create_model(
    const TOKNN_LOSS_FUNCTION loss_function,
    const TOKNN_ACTIVATION_FUNCTION * activation_functions,
    const uint32_t layers_size,
    const uint32_t * nodes_per_layer,
    const uint32_t features_size)
{
    assert(loss_function > TOKNN_LOSS_UNDERFLOW);
    assert(loss_function < TOKNN_LOSS_OVERFLOW);
    assert(layers_size > 0);
    assert(layers_size <= TOKNN_MAX_LAYERS);
    assert(nodes_per_layer != NULL);
    assert(nodes_per_layer[0] > 0);
    for (uint32_t _ = 0; _ < layers_size; _++) {
        assert(activation_functions[_] > TOKNN_ACTIVATION_UNDERFLOW);
        assert(activation_functions[_] < TOKNN_ACTIVATION_OVERFLOW);
    }
    
    TOKNeuralNet * nn = &cache->nn;
    memset(nn, 0, sizeof(TOKNeuralNet));
    
    nn->layers_size = layers_size;
    
    nn->loss_function = loss_function;
    for (uint32_t i = 0; i < layers_size; i++) {
        nn->nodes_per_layer[i] = nodes_per_layer[i];
        assert(nn->nodes_per_layer[i] > 0);
        assert(nn->nodes_per_layer[i] < TOKNN_MAX_NODES);
        nn->activation_functions[i] = activation_functions[i];
    }
    
    nn->features_size = features_size;
    
    nn->nodes_size = to_overall_node_i(
        /* const uint32_t in_layer_i: */
            nn->layers_size-1,
        /* const uint32_t in_layer_node_i: */
            0) + 1;
    
    nn->weights_size = to_overall_weight_i(
        /* const uint32_t in_layer_i: */
            nn->layers_size-1,
        /* const uint32_t in_layer_node_i: */
            0) + (nn->layers_size < 2 ?
                nn->features_size :
                nn->nodes_per_layer[nn->layers_size-2]) + 1;
    
    // initialize weights & biases
    for (uint32_t i = 0; i < nn->weights_size; i++) {
        nn->weights[i] = (float)(i % 3) - 0.9f;
    }
    for (uint32_t i = 0; i < nn->nodes_size; i++) {
        nn->biases[i] = 0.1f;
    }
    
    assert(nn->weights_size >= nn->nodes_size);
    assert(nn->nodes_size > nn->nodes_per_layer[0]);
}

static float toknn_predict_with_nn(
    TOKNeuralNet * nn,
    const float * new_observations,
    const uint32_t new_observations_size)
{
    assert(nn->layers_size > 0);
    assert(nn->loss_function > TOKNN_LOSS_UNDERFLOW);
    assert(nn->loss_function < TOKNN_LOSS_OVERFLOW);
    assert(nn->layers_size > 0);
    assert(nn->nodes_per_layer != NULL);
    assert(nn->nodes_per_layer[0] > 0);
    
    memset(
        nn->activated_node_values,
        0,
        sizeof(float) * TOKNN_MAX_NODES);
    memset(
        nn->preactivated_node_values,
        0,
        sizeof(float) * TOKNN_MAX_NODES);
    
    for (
        uint32_t layer_i = 0;
        layer_i < nn->layers_size;
        layer_i++)
    {
        assert(nn->nodes_per_layer[layer_i] > 0);
        uint32_t previous_layer_start_i = 0;
        if (layer_i > 0) {
            previous_layer_start_i = to_overall_node_i(
                /* const uint32_t in_layer_i: */
                    layer_i-1,
                /* const uint32_t in_layer_node_i: */
                    0);
        }
        
        const float * inputs = layer_i == 0 ?
            new_observations :
            nn->activated_node_values + previous_layer_start_i;
        const uint32_t inputs_size = layer_i == 0 ?
            nn->features_size :
            nn->nodes_per_layer[layer_i-1];
        
        for (
            uint32_t in_layer_node_i = 0;
            in_layer_node_i < nn->nodes_per_layer[layer_i];
            in_layer_node_i++)
        {
            uint32_t node_i = to_overall_node_i(
                /* const uint32_t in_layer_i: */
                    layer_i,
                /* const uint32_t in_layer_node_i: */
                    in_layer_node_i);
            uint32_t first_weight_i = to_overall_weight_i(
                /* const uint32_t in_layer_i: */
                    layer_i,
                /* const uint32_t in_layer_node_i: */
                    in_layer_node_i);
            
            float weighted_sum = nn->biases[node_i];
            for (
                uint32_t input_i = 0;
                input_i < inputs_size;
                input_i++)
            {
                weighted_sum +=
                    nn->weights[first_weight_i + input_i]  *
                    inputs[input_i];
            }
            nn->preactivated_node_values[node_i] = weighted_sum;
            
            switch (nn->activation_functions[layer_i]) {
                case TOKNN_ACTIVATION_NONE:
                    nn->activated_node_values[node_i] =
                        nn->preactivated_node_values[node_i];
                    break;
                case TOKNN_ACTIVATION_TANH:
                    nn->activated_node_values[node_i] =
                        activation_tanh(
                            nn->preactivated_node_values[node_i]);
                    break;
                case TOKNN_ACTIVATION_DOUBLECLAMP_10000:
                    nn->activated_node_values[node_i] =
                        activation_doublecap_10000(
                            nn->preactivated_node_values[node_i]);
                    break;
                case TOKNN_ACTIVATION_DOUBLECLAMP_5000:
                    nn->activated_node_values[node_i] =
                        activation_doublecap_5000(
                            nn->preactivated_node_values[node_i]);
                    break;
                case TOKNN_ACTIVATION_DOUBLECLAMP_2000:
                    nn->activated_node_values[node_i] =
                        activation_doublecap_2000(
                            nn->preactivated_node_values[node_i]);
                    break;
                case TOKNN_ACTIVATION_DOUBLECLAMP_1000:
                    nn->activated_node_values[node_i] =
                        activation_doublecap_1000(
                            nn->preactivated_node_values[node_i]);
                case TOKNN_ACTIVATION_DOUBLECLAMP_200:
                    nn->activated_node_values[node_i] =
                        activation_doublecap_200(
                            nn->preactivated_node_values[node_i]);
                    break;
                default:
                    assert(0);
            }
        }
    }
    
    uint32_t last_layer_i = nn->layers_size - 1;
    assert(nn->nodes_per_layer[last_layer_i] == 1);
    uint32_t output_node_i = to_overall_node_i(
        /* const uint32_t in_layer_i: */
            last_layer_i,
        /* const uint32_t in_layer_node_i: */
            0);
    return nn->activated_node_values[output_node_i];
}

float toknn_predict(
    const float * new_observations,
    const uint32_t new_observations_size)
{
    TOKNeuralNet * nn = &cache->nn;
    
    return toknn_predict_with_nn(
        /* const TOKNeuralNet * nn: */
            nn,
        /* const float * new_observations: */
            new_observations,
        /* const uint32_t new_observations_size: */
            new_observations_size);
}

static void toknn_train_single_epoch(
    TOKNeuralNet * nn,
    const float learning_rate,
    const float * train,
    const uint32_t train_size,
    const float * labels,
    const uint32_t labels_size)
{
    memset(
        cache->optim.all_obs_node_and_bias_derivatives,
        0,
        sizeof(float) * nn->nodes_size);
    memset(
        cache->optim.all_obs_weight_derivatives,
        0,
        sizeof(float) * nn->weights_size);
    
    for (
        uint32_t label_i = 0;
        label_i < labels_size;
        label_i += 1)
    {
        memset(
           cache->optim.this_obs_node_and_bias_derivatives,
           0,
           sizeof(float) * nn->nodes_size);
        memset(
           cache->optim.this_obs_weight_derivatives,
           0,
           sizeof(float) * nn->weights_size);
        
        uint32_t train_i = label_i * nn->features_size;
        
        float deriv_loss = 1;
        
        float yhat = toknn_predict(
            /* const float * new_observations: */
                train + train_i,
            /* const uint32_t new_observations_size: */
                nn->features_size);
        
        switch (nn->loss_function) {
            case TOKNN_LOSS_SQUAREDERROR:
                deriv_loss = derivative_squarederror(
                    /* const float y: */
                        yhat,
                    /* const float t: */
                        labels[label_i]);
                break;
            case TOKNN_LOSS_ABSOLUTEERROR:
                deriv_loss = derivative_absoluteerror(
                    /* const float y: */
                        yhat,
                    /* const float t: */
                        labels[label_i]);
                break;
            default:
                assert(0);
        }
        
        for (
            int32_t layer_i = nn->layers_size-1;
            layer_i >= 0;
            layer_i--)
        {
            for (
                uint32_t inner_node_i = 0;
                inner_node_i < nn->nodes_per_layer[layer_i];
                inner_node_i++)
            {
                uint32_t node_i = to_overall_node_i(
                    /* const uint32_t in_layer_i: */
                        layer_i,
                    /* const uint32_t in_layer_node_i: */
                        inner_node_i);
                
                assert(
                    cache->optim.this_obs_node_and_bias_derivatives[node_i] ==
                        0.0f);
                
                // We want to calculate the weighted derivative of all of
                // the parents our node connects to in the layer above us
                // if we're the output layer, this is just the loss
                // function's derivative
                float layer_above_derivative = 0;
                if (layer_i == nn->layers_size-1) {
                    layer_above_derivative = deriv_loss;
                } else {
                    
                    for (
                        uint32_t parent_inner_node_i = 0;
                        parent_inner_node_i < nn->nodes_per_layer[layer_i+1];
                        parent_inner_node_i += 1)
                    {
                        uint32_t parent_node_i = to_overall_node_i(
                            /* const uint32_t in_layer_i: */
                                layer_i+1,
                            /* const uint32_t in_layer_node_i: */
                                parent_inner_node_i);
                        
                        // we add inner_node_i because we want the weight
                        // that originates from our node to this parent node
                        uint32_t parent_weight_i =
                            to_overall_weight_i(
                                /* const uint32_t in_layer_i: */
                                    layer_i + 1,
                                /* const uint32_t in_layer_node_i: */
                                    parent_inner_node_i) + inner_node_i;
                        
                        layer_above_derivative +=
                            cache->optim.this_obs_node_and_bias_derivatives[
                                parent_node_i] *
                            nn->weights[parent_weight_i];
                    }
                }
                
                float cur_derivative = 0.0f;
                
                switch (nn->activation_functions[layer_i]) {
                     case TOKNN_ACTIVATION_NONE:
                        cur_derivative = 1.0f;
                        break;
                     case TOKNN_ACTIVATION_TANH:
                        cur_derivative = derivative_tanh(
                                nn->preactivated_node_values[node_i]);
                        break;
                     case TOKNN_ACTIVATION_DOUBLECLAMP_10000:
                        cur_derivative = derivative_doublecap_10000(
                                nn->preactivated_node_values[node_i]);
                        break;
                     case TOKNN_ACTIVATION_DOUBLECLAMP_5000:
                        cur_derivative = derivative_doublecap_5000(
                                nn->preactivated_node_values[node_i]);
                        break;
                     case TOKNN_ACTIVATION_DOUBLECLAMP_2000:
                        cur_derivative = derivative_doublecap_2000(
                                nn->preactivated_node_values[node_i]);
                        break;
                     case TOKNN_ACTIVATION_DOUBLECLAMP_1000:
                        cur_derivative = derivative_doublecap_1000(
                                nn->preactivated_node_values[node_i]);
                        break;
                     case TOKNN_ACTIVATION_DOUBLECLAMP_200:
                        cur_derivative = derivative_doublecap_200(
                                nn->preactivated_node_values[node_i]);
                        break;
                     default:
                        assert(0);
                }
                
                float node_deriv = layer_above_derivative * cur_derivative;
                
                cache->optim.this_obs_node_and_bias_derivatives[node_i] =
                    node_deriv;
                
                uint32_t first_weight_i = to_overall_weight_i(
                    layer_i, inner_node_i);
                
                const float * inputs =
                    layer_i == 0 ?
                        train + train_i :
                        nn->activated_node_values +
                            to_overall_node_i(layer_i-1, 0);
                uint32_t inputs_size =
                    layer_i == 0 ?
                        nn->features_size :
                        nn->nodes_per_layer[layer_i-1];
                
                for (
                    uint32_t input_i = 0;
                    input_i < inputs_size;
                    input_i++)
                {
                    cache->optim.this_obs_weight_derivatives[
                        first_weight_i + input_i] =
                            cache->optim.
                                this_obs_node_and_bias_derivatives[node_i] *
                                    inputs[input_i];
                }
            }
        }
        
        for (uint32_t i = 0; i < nn->weights_size; i++) {
            cache->optim.all_obs_weight_derivatives[i] +=
                (cache->optim.this_obs_weight_derivatives[i] /
                    labels_size);
        }
        for (uint32_t i = 0; i < nn->nodes_size; i++) {
            cache->optim.all_obs_node_and_bias_derivatives[i] +=
                (cache->optim.this_obs_node_and_bias_derivatives[i] /
                    labels_size);
        }
        
        #ifndef NDEBUG
        TOKNeuralNet * nndb = &cache->nn_debug;
        *nndb = *nn;
        
        float current_yhat = toknn_predict(
            /* const float * new_observations: */
                train + (label_i * nndb->features_size),
            /* const uint32_t new_observations_size: */
                nndb->features_size);
        float current_loss = loss_absoluteerror(
            current_yhat,
            labels[label_i]);
        
        for (
            int32_t layer_i = nndb->layers_size-1;
            layer_i >= 0;
            layer_i--)
        {
            float nudge = 0.001f;
            for (
                 uint32_t node_i = 0;
                 node_i < nndb->nodes_per_layer[layer_i];
                 node_i++)
            {
                uint32_t weight_i =
                    to_overall_weight_i(layer_i, node_i);
                
                nndb->weights[weight_i] += nudge;
                float nudged_yhat = toknn_predict_with_nn(
                        nndb,
                    /* const float * new_observations: */
                        train + (label_i * nndb->features_size),
                    /* const uint32_t new_observations_size: */
                        nndb->features_size);
                float nudged_loss = loss_absoluteerror(
                    nudged_yhat,
                    labels[label_i]);
                nndb->weights[weight_i] -= nudge;
                
                float numerical_deriv =
                    (nudged_loss - current_loss) / nudge;
                
                float tolerance = 0.05f;
                if (
                    fabs(current_yhat) > 0.03f &&
                    fabs(numerical_deriv -
                         cache->optim.this_obs_weight_derivatives[weight_i]) >
                    tolerance)
                {
                    assert(0);
                }
            }
            
            for (
                uint32_t inner_node_i = 0;
                inner_node_i < nndb->nodes_per_layer[layer_i];
                inner_node_i++)
            {
                uint32_t bias_i = to_overall_node_i(
                    /* const uint32_t in_layer_i: */
                        layer_i,
                    /* const uint32_t in_layer_node_i: */
                        inner_node_i);
                
                nndb->biases[bias_i] += nudge;
                float nudged_yhat = toknn_predict_with_nn(
                        nndb,
                    /* const float * new_observations: */
                        train + (label_i * nndb->features_size),
                    /* const uint32_t new_observations_size: */
                        nndb->features_size);
                float nudged_loss = loss_absoluteerror(
                    nudged_yhat,
                    labels[label_i]);
                
                nndb->biases[bias_i] -= nudge; // revert nudge
                
                float numerical_deriv =
                    (nudged_loss - current_loss) / nudge;
                
                float tolerance = 0.05f;
                if (
                    fabs(current_yhat) > 0.03f &&
                    fabs(numerical_deriv -
                        cache->optim.this_obs_node_and_bias_derivatives[bias_i]) >
                            tolerance)
                {
                     assert(0);
                }
            }
        }
        #endif
    }
    
    // the final layer is guaranteed to have only 1 node, so this is
    // guaranteed to be past it
    for (uint32_t i = 0; i < nn->weights_size; i++) {
        float avg_deriv = cache->optim.all_obs_weight_derivatives[i];
        float reduction = learning_rate * avg_deriv;
        nn->weights[i] -= reduction;
    }
    for (uint32_t i = 0; i < nn->nodes_size; i++) {
        float avg_deriv = cache->optim.all_obs_node_and_bias_derivatives[i];
        float reduction = learning_rate * avg_deriv;
        nn->biases[i] -= reduction;
    }
}

static void nn_report_progress(
    const TOKNeuralNet * nn,
    const uint32_t epoch_i,
    const float * train,
    const uint32_t train_size,
    const float * labels,
    const uint32_t labels_size)
{
    float loss_sum = 0.0f;
    
    for (uint32_t label_i = 0; label_i < labels_size; label_i++) {
        float new_yhat = toknn_predict(
            /* const float * new_observations: */
                train + (label_i * nn->features_size),
            /* const uint32_t new_observations_size: */
                nn->features_size);
        
        switch (nn->loss_function) {
            case TOKNN_LOSS_SQUAREDERROR:
                loss_sum += (float)loss_squarederror(
                    /* const float y: */
                        new_yhat,
                    /* const float t: */
                        labels[label_i]) / (float)labels_size;
                break;
            case TOKNN_LOSS_ABSOLUTEERROR:
                loss_sum += (float)loss_absoluteerror(
                    /* const float y: */
                        new_yhat,
                    /* const float t: */
                        labels[label_i]) / (float)labels_size;
                break;
            default:
                assert(0);
        }
    }
    
    printf("Epoch: %u Loss: %f\n", epoch_i, loss_sum);
}

void toknn_train(
    const float * train,
    const uint32_t train_size,
    const float * labels,
    const uint32_t labels_size)
{
    TOKNeuralNet * nn = &cache->nn;
    assert(nn->layers_size > 0);
    assert(nn->nodes_per_layer[nn->layers_size-1] == 1);
    assert(nn->features_size > 0);
    assert(train_size >= labels_size);
    assert(train_size / labels_size == nn->features_size);
    
    float learning_rate = 0.011f;
    
    for (uint32_t epoch_i = 0; epoch_i < 10000; epoch_i++) {
        
        learning_rate *= 0.999f;
        
        toknn_train_single_epoch(
            /* TOKNeuralNet * net: */
                nn,
            /* const float learning_rate: */
                learning_rate,
            /* const float * train: */
                train,
            /* const uint32_t train_size: */
                train_size,
            /* const float * labels: */
                labels,
            /* const uint32_t labels_size: */
                labels_size);
        
        // Report progress
        if (epoch_i < 10 || epoch_i % 20 == 0) {
            nn_report_progress(
               /* const TOKNeuralNet * nn: */
                   nn,
                /* const uint32_t epoch_i: */
                    epoch_i,
               /* const float * train: */
                   train,
               /* const uint32_t train_size: */
                   train_size,
               /* const float * labels: */
                   labels,
               /* const uint32_t labels_size: */
                   labels_size);
        }
    }
}

