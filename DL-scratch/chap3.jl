using LogExpFunctions
using Plots

## step関数の実装
function step_fuction(x)
    if x > 0
        return 1
    else
        return 0
    end
end

function step_fuction(x)
    return float(map(i -> i > 0.0, x))
end

x = range(-5, 5, step=0.1)
step_fuction(x)

plot(x, step_fuction(x))

## シグモイド関数の実装
function sigmoid_function(x)
    return float(map(i -> 1 / (1 + exp(-i)), x))
end

sigmoid_function(x)
plot(x, sigmoid_function(x))


## ReLu関数の実装
function relu_function(x)
    return map(i -> max(0, i), x)
end

relu_function(x)
plot(x, relu_function(x))