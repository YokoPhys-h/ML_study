# cos(x)をReLu関数を用いた一層NNで近似します.
using Plots
using LinearAlgebra
using ForwardDiff
using ForwardDiff: gradient

n = 5 # 中間層サイズ

## 対象関数を教師情報の代わりに作成
predict(x) = cos(x)

##グラフのチェック
X = range(0, 2.0 * π, step=0.01)
plot(X, x -> predict(cos(x)))

## NNの作成
### ReLu関数定義
relu(x) = (x > 0.0) ? x : 0.0

# Neural Network. 
function nn(M, input)
    C1 = view(M, :, 1)
    b1 = view(M, :, 2)

    C2 = view(M, :, 3:n+2)
    b2 = view(M, :, n + 3)

    C3 = view(M, :, n + 4)
    b3 = M[1, n+5]

    r1 = relu.(input * C1 + b1) # 中間層1 出力
    r2 = relu.(C2 * r1 + b2)    # 中間層2 出力
    r3 = dot(C3, r2) + b3    # 出力

    return r3

end

## 損失関数 とりあえず二乗誤差
loss(x, y) = (x - y)^2

## NN のパラメータ行列 H の初期値を乱数で生成
H = rand(n, n + 5) .- 0.5

## 学習前の初期関数の様子
plot(X, x -> nn(H, x))

##########

# 計算開始

f(H, x) = @inbounds loss(predict(x), nn(H, x))

@showprogress for i in 1:200000  # 200000個のデータ
    x = rand()  # ランダムなインプットx

    grad_f = @inbounds gradient(H -> f(H, x), H) # 誤差Hに対する勾配

    grad_size = norm(grad_f)^2 # 勾配の大きさ

    if grad_size < 0.05 # 学習係数の調整(数字は適当)
        γ = 0.1
    else
        γ = f(H, x) / grad_size
    end

    H += -γ * grad_f # パラメータ H を修正．
end

## 学習によるplot
plot(X, x -> nn(H, x))
H