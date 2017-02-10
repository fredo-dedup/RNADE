using VegaLite
using Distributions
using ReverseDiffSource


xs = vcat(rand(Normal(-2,1.),500), rand(Normal(2,1.),500))
data_values(x=xs) + encoding_x_quant(:x) + mark_tick()


####################################################

N = 3
w₁ = rand(Normal(), N)
b₁ = rand(Normal(), N)
w₂ = rand(Normal(), N)

oact(x) = 1 / (1+exp(-x))
f(x::Float64) = dot( oact.(w₂ .* x + b₁), w₂) / sum(w₂)

ec = quote
    cw1 = log(1. .+ exp(w₁))
    cw2 = log(1. .+ exp(w₂))
    cw2 = cw2 ./ sum(cw2)
    dot( 1. ./ ( 1. + exp(- cw1 .* (x .+ b₁))), cw2 )
end

ep = rdiff(ec, allorders=false, x=Float64)
@eval fp(x) = $ep  # prob function
@eval fl(x) = log($ep)  # ll function

fp(2.)
fl(2.)

edl = rdiff(:(log($ep)), allorders=false, w₁=Vector{Float64}, b₁=Vector{Float64}, w₂=Vector{Float64})
@eval fdl(x) = $edl # diffs of loglik function

fdl(2.)



function fulldp(xs::Vector{Float64})
    dw₁ = zeros(w₁)
    db₁ = zeros(b₁)
    dw₂ = zeros(w₂)
    for x in xs
      δw₁, δb₁, δw₂ = fdl(x)
      dw₁ += δw₁
      db₁ += δb₁
      dw₂ += δw₂
    end
    dw₁,db₁,dw₂
end

fulldp(xs)



N = 4
w₁ = ones(N)
b₁ = linspace(-5,5, N)
w₂ = ones(N)
mean( fl(x) for x in xs )

μ = 1e-1
    dw₁, db₁, dw₂ = fulldp(xs)
    ρ = 1/max(norm(dw₁), norm(db₁), norm(dw₂))
    w₁ += μ * min(1., ρ) * dw₁
    b₁ += μ * min(1., ρ) * db₁
    w₂ += μ * min(1., ρ) * dw₂
    mean( fl(x) for x in xs )

norm(dw₁)


xp = linspace(-10, 10, 1000)
data_values(x=xp, y=map(fp,xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()

[w₁ b₁ w₂]






[dw₁ db₁ dw₂]





####### truncated version

ecden = quote
    cw1 = log(1. .+ exp(w₁))
    cw2 = log(1. .+ exp(w₂))
    cw2 = cw2 ./ sum(cw2)
    dot( 1. ./ ( 1. + exp(- cw1 .* (1.+b₁))) - 1. ./ ( 1. + exp(- cw1 .* b₁)), cw2 )
end


@eval fc(x)
@eval fp(x) = $ep / $ecden  # prob function
@eval fl(x) = log($ep / $ecden)  # ll function

@eval $ecden

fp(0.5)
fl(2.)

edl = rdiff(:(log($ep/$ecden)), allorders=false, w₁=Vector{Float64}, b₁=Vector{Float64}, w₂=Vector{Float64})
@eval fdl(x) = $edl # diffs of loglik function

fdl(2.)



function fulldp(xs::Vector{Float64})
    dw₁ = zeros(w₁)
    db₁ = zeros(b₁)
    dw₂ = zeros(w₂)
    for x in xs
      δw₁, δb₁, δw₂ = fdl(x)
      dw₁ += δw₁
      db₁ += δb₁
      dw₂ += δw₂
    end
    dw₁,db₁,dw₂
end

fulldp(xs)
xs = rand(Beta(3., 2.), 1000)
extrema(xs)

N = 10
w₁ = ones(N)
b₁ = linspace(0,1, N)
w₂ = ones(N)
mean( fl(x) for x in xs )

μ = 1e-0
    dw₁, db₁, dw₂ = fulldp(xs)
    ρ = 1/max(norm(dw₁), norm(db₁), norm(dw₂))
    w₁ += μ * min(1., ρ) * dw₁
    b₁ += μ * min(1., ρ) * db₁
    w₂ += μ * min(1., ρ) * dw₂
    mean( fl(x) for x in xs )

norm(dw₁)

fp(0.5)
xp = collect(linspace(0, 1, 1000))
data_values(x=xp, y=map(fp,xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()
data_values(x=xp, y=pdf(Beta(3., 2.),xp)) + encoding_x_quant(:x) + encoding_y_quant(:y) + mark_line()

[w₁ b₁ w₂]










[dw₁ db₁ dw₂]
