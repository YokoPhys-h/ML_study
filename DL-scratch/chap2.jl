## perceptronの実装
function AND(x1, x2)
    w1 = 0.5
    w2 = 0.5
    θ = 0.7
    temp = x1 * w1 + x2 * w2
    if temp <= θ
        return 0
    elseif temp > θ
        return 1
    end
end

AND(1, 1)

x = [0, 1]
w = [0.5, 0.5]
b = -0.7

sum(w .* x) + b

# 重みとバイアスによる実装
function AND_2(x1, x2)
    x = [x1, x2]
    w = [0.5, 0.5]
    b = -0.7
    temp = sum(w .* x) + b
    if temp <= 0
        return 0
    elseif temp > 0
        return 1
    end
end

AND_2(1, 1)
