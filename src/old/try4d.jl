################################################################
##  égal à tr3d.jl avec composante centrée sur la valeur précédente
################################################################

using Distributions
# using BenchmarkTools
using VegaLite


include("setup3.jl")

# Mixture de Kumaraswamy

Nh = 50  # number of hidden units
Ne = 5   # number of mixture elements
Nd = 50  # variable size to estimate

sigm(x::Float64) = 1 ./ (1+exp(-x))

type Pars
  Vm::Vector{Matrix{Float64}}  # for m mixture component
  Vn::Vector{Matrix{Float64}}  # for n mixture component
  Vw::Vector{Matrix{Float64}}  # for weight mixture component
  bm::Vector{Vector{Float64}}  # for m mixture component
  bn::Vector{Vector{Float64}}  # for n mixture component
  bw::Vector{Vector{Float64}}  # for weight mixture component
  W::Matrix{Float64}
  c::Vector{Float64}
end

function add!(a::Pars, b::Pars)
  for i in 1:Ne
    a.Vm[i] .+= b.Vm[i]
    a.Vn[i] .+= b.Vn[i]
    a.Vw[i] .+= b.Vw[i]
    a.bm[i] .+= b.bm[i]
    a.bn[i] .+= b.bn[i]
    a.bw[i] .+= b.bw[i]
  end
  a.W .+= b.W
  a.c .+= b.c
  a
end

function scal!(a::Pars, f::Float64)
  for i in 1:Ne
    a.Vm[i] .*= f
    a.Vn[i] .*= f
    a.Vw[i] .*= f
    a.bm[i] .*= f
    a.bn[i] .*= f
    a.bw[i] .*= f
  end
  a.W .*= f
  a.c .*= f
  a
end

function zeros!(a::Pars)
  for i in 1:Ne
    fill!(a.Vm[i], 0.)
    fill!(a.Vn[i], 0.)
    fill!(a.Vw[i], 0.)
    fill!(a.bm[i], 0.)
    fill!(a.bn[i], 0.)
    fill!(a.bw[i], 0.)
  end
  fill!(a.W, 0.)
  fill!(a.c, 0.)
  a
end


function Pars(Nh, Nd, Ne)
    Pars([rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nh, Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         [rand(Normal(), Nd) for i in 1:Ne],
         rand(Normal(), Nh, Nd),
         rand(Normal(), Nh)
    )
end


function loglik(pars::Matrix{Float64}, x::Float64, x₀::Float64)
    ws = exp.(pars[:,3])
    ws ./= sum(ws)
    # pm = dot(ws[1:end-1], fk3.(x, pars[1:end-1,1], pars[1:end-1,2]))
    pm = 0.
    for i in 1:length(ws)-1
      pm += ws[i] * fk3(x, pars[i,1], pars[i,2])
    end
    pm += ws[end] * pdf(Normal(x₀, 0.01), x)
    -log(pm)
end
# loglik(pars, 0.5, 0.5)
# exp(-loglik(pars, 0.1))
#
# @benchmark loglik(pars, 0.5) # 28us


# delta = 1e-4
# pars1 = copy(pars) ; pars1[1,1] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,2] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# pars1 = copy(pars) ; pars1[1,3] += delta
# (loglik(pars1,0.9)-loglik(pars,0.9))/delta
#
# dloglik(pars, 0.9)[6,:]


function dloglik!(pars::Matrix{Float64}, dpars::Matrix{Float64},
                  x::Float64, x₀::Float64)
    # p = 0.5
    ne = size(pars,1)
    ws0 = exp.(pars[:,3])
    sws0 = sum(ws0)
    ws = ws0 ./ sws0
    ps = Array(Float64, ne)
    # ps[1:ne-1] = fk3.(x, pars[1:ne-1,1], pars[1:ne-1,2])
    for i in 1:ne-1
      ps[i] = fk3(x, pars[i,1], pars[i,2])
    end
    ps[ne] = pdf(Normal(x₀, 0.01), x)
    # ps = vcat(fk3.(x, pars[1:end-1,1], pars[1:end-1,2]),
    #           pdf(Normal(x₀, 0.01), x) )

    pm = dot(ws, ps)

    _tmp5 = - ps ./ pm
    dpars[:,3] = ws0 ./ sws0 .* ( _tmp5 + ( sum(-ws0 .* _tmp5) / sws0 ) )

    dv0 = -1/pm
    for i in 1:ne-1
        dpars[i,1], dpars[i,2] = fdk3(x, pars[i,1], pars[i,2])
        dpars[i,1] *= ws[i] * dv0
        dpars[i,2] *= ws[i] * dv0
    end
    dpars[ne,1], dpars[ne,2] = 0., 0.

    dpars
end

# ipars = rand(Normal(),5,3)
#
# loglik(ipars, 0.5, 0.4)
# dipars = zeros(ipars)
# dloglik!(ipars, dipars, 0.5, 0.4)
#
# ipars2 = copy(ipars); ipars2[2,2] += δ
# (loglik(ipars2, 0.5, 0.4)-loglik(ipars, 0.5, 0.4)) / δ
# dipars[2,2]
#
# ipars2 = copy(ipars); ipars2[5,2] += δ
# (loglik(ipars2, 0.5, 0.4)-loglik(ipars, 0.5, 0.4)) / δ
# dipars[5,2]
#
# ipars2 = copy(ipars); ipars2[5,1] += δ
# (loglik(ipars2, 0.5, 0.4)-loglik(ipars, 0.5, 0.4)) / δ
# dipars[5,1]
#
# ipars2 = copy(ipars); ipars2[5,3] += δ
# (loglik(ipars2, 0.5, 0.4)-loglik(ipars, 0.5, 0.4)) / δ
# dipars[5,3]

pars = Pars(Nh, Nd, Ne)
dpars = deepcopy(pars)
whos(r"pars") # ~400kb

xs = rand(20)


function xloglik(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
  xt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  cpars = Array(Float64, Ne, 3)
  for i in 1:nx # i = 1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    xt[i] = loglik(cpars, xs[i], (i==1) ? xs[1] : xs[i-1])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs[i]
  end
  ll, xt, h
end

xloglik(xs, pars)


function xdloglik!(xs::Vector{Float64}, pars::Pars, dpars::Pars) # x, pars = x₀, pars₀
  nx = length(xs)
  ll, xt, h = xloglik(xs, pars)

  zeros!(dpars)

  δh = zeros(h)
  δa = zeros(pars.c)

  cpars = Array(Float64, Ne, 3)
  dcpars = similar(cpars)
  for i in nx:-1:1 # i = nx
    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    dloglik!(cpars, dcpars, xs[i], (i==1) ? xs[1] : xs[i-1])

    for j in 1:Ne
      local m,n,w
      dm,dn,dw = dcpars[j,1], dcpars[j,2], dcpars[j,3]
      dpars.bm[j][i] = dm
      dpars.bn[j][i] = dn
      dpars.bw[j][i] = dw

      dpars.Vm[j][:,i] = dm .* h[:,i]
      dpars.Vn[j][:,i] = dn .* h[:,i]
      dpars.Vw[j][:,i] = dw .* h[:,i]

      δh[:,i] += pars.Vm[j][:,i] .* dm +
                 pars.Vn[j][:,i] .* dn +
                 pars.Vw[j][:,i] .* dw
    end

    dpars.W[:,i] = δa * xs[i]
    δa     += δh[:,i] .* h[:,i] .* (1. - h[:,i])
  end
  copy!(dpars.c, δa)
  dpars
end

xloglik(xs, pars)
xdloglik!(xs, pars, dpars)
dpars.c[1]
dpars.c

δ = 1e-8

###############  profiling #################

Profile.clear()
Base.@profile collect(xdloglik!(dat[:,i], pars, dpars) for i in 50:500)
Profile.print()

@time collect(xdloglik!(dat[:,i], pars, dpars) for i in 50:500)
# 3.83 s
# 3.14 s
# 3.00 s
# 2.86 s
# 1.81 s
# 0.71 s


###############  testing ########################

###########   c   ok
pars2 = deepcopy(pars)
pars2.c[1]
pars2.c[1] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[1]

pars2 = deepcopy(pars)
pars2.c[20]
pars2.c[20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[20]

pars2 = deepcopy(pars)
pars2.c[40]
pars2.c[40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.c[40]

############### W ok
pars2 = deepcopy(pars)
pars2.W[10,10]
pars2.W[10,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.W[10,10]


pars2 = deepcopy(pars)
pars2.W[40,10]
pars2.W[40,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.W[40,10]

##########  b
# bn OK (convexité élevée !)
pars2 = deepcopy(pars)
pars2.bn[2][20]
pars2.bn[2][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[2][20]

pars2 = deepcopy(pars)
pars2.bn[1][4]
pars2.bn[1][4] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[1][4]

pars2 = deepcopy(pars)
pars2.bn[5][35]
pars2.bn[5][35] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bn[5][35]

# bw , OK
pars2 = deepcopy(pars)
pars2.bw[2][30]
pars2.bw[2][30] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][30]

pars2 = deepcopy(pars)
pars2.bw[3][6]
pars2.bw[3][6] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[3][6]

pars2 = deepcopy(pars)
pars2.bw[2][20]
pars2.bw[2][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][20]


pars2 = deepcopy(pars)
pars2.bw[2][13]
pars2.bw[2][13] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bw[2][13]

# bm, OK
pars2 = deepcopy(pars)
pars2.bm[5][1]
pars2.bm[5][1] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[5][1]

pars2 = deepcopy(pars)
pars2.bm[4][20]
pars2.bm[4][20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[4][20]

pars2 = deepcopy(pars)
pars2.bm[4][40]
pars2.bm[4][40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.bm[4][40]


########  W  OK
# Ww, ok
pars2 = deepcopy(pars)
pars2.Vw[2][5,10]
pars2.Vw[2][5,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vw[2][5,10]

# Wm, ok
pars2 = deepcopy(pars)
pars2.Vw[5][15,10]
pars2.Vw[5][15,10] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vw[5][15,10]

# Wn, ok
pars2 = deepcopy(pars)
pars2.Vn[1][20,20]
pars2.Vn[1][20,20] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vn[1][20,20]


# Wn, ok
pars2 = deepcopy(pars)
pars2.Vn[1][20,40]
pars2.Vn[1][20,40] += δ

(xloglik(xs,pars2)[1]-xloglik(xs,pars)[1]) / δ
dpars.Vn[1][20,40]
