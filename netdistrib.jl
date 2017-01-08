using VegaLite
using Distributions
using ReverseDiffSource


N = 10
w₁ = rand(N)
w₂ = rand(N)

act(x)  = log(1 + exp(x))
sigm(x) = 1 / (1+exp(-x))
f(x::Float64) = sigm( dot(w₂, act.(w₁ * x) ) )

f(3.)

f.( collect(-2:0.1:2.0 ))

xp = linspace(-2, 2, 1000)
data_values(x=xp, y=f.(xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()


xs = rand(Normal(),1000)

function pdiff()
  v0 = sum( -log(f(x)) for x in xs )
  δ = 1e-8
  dw₁ = similar(w₁)
  for i in 1:N
    w₁[i] += δ
    dw₁[i] = (sum( -log(f(x)) for x in xs ) - v0) / δ
    w₁[i] -= δ
  end
  dw₂ = similar(w₂)
  for i in 1:N
    w₂[i] += δ
    dw₂[i] = (sum( -log(f(x)) for x in xs ) - v0) / δ
    w₂[i] -= δ
  end
  dw₁,dw₂
end


μ = 1e-3
ρ =

w₁ = rand(Normal(),N) ; w₂ = rand(Normal(),N)

dw₁,dw₂ = pdiff()
w₁ += μ * dw₁ ; w₂ += μ * dw₂
sum( -log(f(x)) for x in xs )


############# CDF contraint


N = 10
w₁ = vcat(1., -1., 0.5*rand(N-2))
w₂ = vcat(1., -1., 0.5*rand(N-2))

iact(x) = log(1 + exp(x))
oact(x) = 1 / (1+exp(-x))
f(x::Float64) = oact( dot(w₂, iact.(w₁ * x) ) )

δ = 1e-8
f(1.)
(f(1.+δ)-f(1.))/δ

edf = rdiff(:( 1 / (1+exp(- dot(w₂, log(1. + exp(w₁ * x) ) ) )) ), allorders=false,
            x=Float64, w₁=Vector{Float64}, w₂=Vector{Float64}, ignore=[:w₁, :w₂] )

@eval df(x) = $edf
df(1.)

f.( collect(-2:0.1:2.0 ))

xp = linspace(-2, 2, 1000)
data_values(x=xp, y=f.(xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()
data_values(x=xp, y=df.(xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()

xs = rand(Normal(),1000)
xs = vcat(rand(Normal(-1,0.4),500), rand(Normal(1,0.4),500))
data_values(x=xs) + encoding_x_quant(:x) + mark_tick()

df(xs[500])

ll(x) = -log(max(1e-8,df(x)))

function pdiff()
  v0 = sum( ll(x) for x in xs )
  δ = 1e-8
  dw₁ = similar(w₁)
  for i in 1:N
    w₁[i] += δ
    dw₁[i] = (sum( ll(x) for x in xs ) - v0) / δ
    w₁[i] -= δ
  end
  dw₂ = similar(w₂)
  for i in 1:N
    w₂[i] += δ
    dw₂[i] = (sum( ll(x) for x in xs ) - v0) / δ
    w₂[i] -= δ
  end
  dw₁,dw₂
end


μ = 1e-4
ρ =

w₁ = vcat(1., -1., 0.5*rand(N-2))
w₂ = vcat(1., -1., 0.5*rand(N-2))
mean( ll(x) for x in xs )

dw₁,dw₂ = pdiff()
w₁ -= μ * dw₁ ; w₂ -= μ * dw₂
# w₂[1] = max( 0.01, max(w₂[1], maximum(w₂[2:end])    +0.01)) # should be positive
# w₂[2] = min(-0.01, min(w₂[2], minimum(w₂[[1;3:end]])-0.01)) # should be negative
# w₁[1] = max( 0.01, max(w₁[1], maximum(w₁[2:end])    +0.01)) # should be positive
# w₁[2] = min(-0.01, min(w₁[2], minimum(w₁[[1;3:end]])-0.01)) # should be negative

mean( ll(x) for x in xs )
