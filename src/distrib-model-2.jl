################################################################
#
#  using Bernstein polynomials
#  with x0 set at 0.5
#
################################################################

module DModel

using Distributions
# using VegaLite
# using ReverseDiffSource

import Distributions: pdf, logpdf, rand
export logpdf, dlogpdf!, PDist, rand, updatePDist!

# calculates normalized probability weights & jacobian
function _updatePDIST!(uw::Vector{Float64}, w::Vector{Float64}, dw::Matrix{Float64})
  ew = exp(uw)
  ewp1 = 1.0 .+ ew
  w2 = log(ewp1)
  sw2 = sum(w2)
  copy!(w, w2 ./ sw2)

  copy!(ew, ew ./ ewp1 ./ sw2)  # mem reuse
  for i in 1:length(uw)
    for j in 1:length(uw)
      dw[i,j] = ew[i] * ( Float64(i==j) - w[j] )
    end
  end

  # copy!(dw, ew ./ ewp1 ./ sw2 .* ( eye(length(uw)) .- w' ))
end

# distrib type
immutable PDist
  w::Vector{Float64}
  dw::Matrix{Float64}
  binom::Vector{Float64}
  N::Int

  function PDist(uw::Vector{Float64})
    N = length(uw)
    binom = Float64[binomial(N-4,i-1) for i in 1:N-3]
    w = Array(Float64, N)
    dw = Array(Float64, N, N)
    _updatePDIST!(uw, w, dw)

    new(w, dw, binom, N)
  end
end


#  weights update
updatePDist!(d::PDist, uw::Vector{Float64}) = _updatePDIST!(uw, d.w, d.dw)

# pd = PDist([-1., 0., 1., 5.])
# pd.dw * ones(4)
#
# function pdf(d::PDist, x)
#   d.w[2]
# end
#
# δ = 1e-8
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1.+δ, 0., 1., 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0.+δ, 1., 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0., 1.+δ, 5.]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
# uw = [-1., 0., 1., 5.]
# uw2 = [-1., 0., 1., 5.+δ]
# (pdf(PDist(uw2), 4.) - pdf(PDist(uw), 4.)) / δ
#
#
# PDist(uw).dw



############## continuous internal model specs  #############################

### Bernstein polynomials Distribution

### proba
function fp(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
  N = length(w) - 3
  ps = zeros(N)
  vx = 1.
  for i in 1:N
    ps[i] = vx * w[i] * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    ps[i] = ps[i] * vx
    vx = vx * (1. - x)
  end

  sum(ps) * N
end

# uw = [0., -1, 1, 2, 3, -2]
# pd = PDist(uw)
#
# fp(0.5, pd.w, pd.binom)
# fp(0., pd.w, pd.binom)
# fp(1., pd.w, pd.binom)
#
# quadgk(x -> fp(x, pd.w, pd.binom), 0., 1.)
# sum(pd.w[1:3])

### proba + gradient
function dfp!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(w) - 3
  ps = zeros(N)
  vx = 1.
  for i in 1:N
    ps[i] = vx * w[i] * cw[i]
    dps[i] = vx * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    ps[i] *= vx
    dps[i] *= vx * N
    vx = vx * (1. - x)
  end

  sum(ps) * N
end

# dps = zeros(length(pd.w))
# dfp!(0.5, pd.w, pd.binom, dps)
# dps
#
# pd2 = deepcopy(pd); pd2.w[1] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[2] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[3] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[4] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[5] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ
#
# pd2 = deepcopy(pd); pd2.w[6] += δ
#   (fp(0.5, pd2.w, pd2.binom)-fp(0.5, pd.w, pd.binom)) / δ


### gradient only
function dfp2!(x::Float64, w::Vector{Float64}, cw::Vector{Float64}, dps::Vector{Float64})
  N = length(w) - 3
  vx = 1.
  for i in 1:N
    dps[i] = vx * cw[i]
    vx = vx * x
  end

  vx = 1.
  for i in N:-1:1
    dps[i] *= vx * N
    vx = vx * (1. - x)
  end
end



############## full model (continuous part + steps at 0, 1. and x0  #################

# pick a value
function rand(d::PDist, x₀::Float64)
  ci = rand(Categorical(d.w))

  ci == d.N && return 1.   # on 1.
  ci == d.N-1 && return 0.5 # centered on x(t-1)
  ci == d.N-2 && return 0. # on 0.

  0.5
end

#   pdf(x) = x^(ci-1) * (1. - x)^(N-3) * d.binom[ci]
#   function fp(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
#     N = length(w) - 3
#     ps = zeros(N)
#     vx = 1.
#     for i in 1:N
#       ps[i] = vx * w[i] * cw[i]
#       vx = vx * x
#     end
#
#     vx = 1.
#     for i in N:-1:1
#       ps[i] = ps[i] * vx
#       vx = vx * (1. - x)
#     end
#
#     sum(ps) * N
#   end
#
#
#   xs2[i] = ficdf3(vcpars[i][ci], vcpars[i][ci+Ne], rand())
#   end
# end

# ### inverse cumulative density TODO
# function ficdf(x::Float64, w::Vector{Float64}, cw::Vector{Float64})
#
# end

# fcdf3(-5.,10.,0.5)
# fcdf3.(-2.,2.,[0:0.1:1;])

####### loglik definitions w/  #######################################

const pwidth = 1e-2

function logpdf(d::PDist, x::Float64, x₀::Float64)
  # x = 0.35 ; d = pd ; x₀ = 0.2
  x₀ = 0.5
  pm = fp(x, d.w, d.binom)

  pw0, pw1, hpw, ipw = pwidth, 1. - pwidth, pwidth / 2., 1. / pwidth
  x00 = clamp(x₀ - hpw, 0. , pw1)
  x01 = clamp(x₀ + hpw, pw0, 1. )
  x<pw0 && (pm += d.w[end-2]*ipw)
  x00 < x < x01 && (pm += d.w[end-1]*ipw)
  x>pw1 && (pm += d.w[end]*ipw)

  -log(pm)
end

# logpdf(pd, 0.9, 0.5)

# uw = rand(Normal(),4)
# pd = PDist(uw)
# quadgk( x -> exp(-logpdf(pd, x, 0.2)), 0., 1.)
# quadgk( x -> exp(-logpdf(pd, x, 0.)), 0., 1.)


# x, x₀ = 0.5, 0.
function dlogpdf!(d::PDist, x::Float64, x₀::Float64, duw::Vector{Float64})
  x₀ = 0.5
  pm = dfp!(x, d.w, d.binom, duw)

  pw0, pw1, hpw, ipw = pwidth, 1. - pwidth, pwidth / 2., 1. / pwidth
  x00 = clamp(x₀ - hpw, 0. , pw1)
  x01 = clamp(x₀ + hpw, pw0, 1. )

  if (x < pw0)
    duw[end-2] = ipw
    pm += d.w[end-2]*ipw
  else
    duw[end-2] = 0.
  end

  if (x00 < x < x01)
    duw[end-1] = ipw
    pm += d.w[end-1]*ipw
  else
    duw[end-1] = 0.
  end

  if (x > pw1)
    duw[end]   = ipw
    pm += d.w[end]*ipw
  else
    duw[end] = 0.
  end

  copy!(duw, - d.dw * duw ./ pm)
end


#####  testing
if false

δ = 1e-8
function dtest(uw, x, x₀, index)
  # x, x₀, index = 0.5, 0.5, 3
  w0 = getindex(uw, index)
  d0 = PDist(uw)
  uw1 = copy(uw)
  setindex!(uw1, w0+δ, index)
  d1 = PDist(uw1)

  ed = (logpdf(d1, x, x₀) - logpdf(d0, x, x₀)) / δ

  duw = Array(Float64, length(uw))
  dlogpdf!(d0, x, x₀, duw)
  ed0 = getindex(duw, index)
  ( ed0, ed )
end

uw = rand(Normal(), 6)

dtest(uw, 0.5, 0.4, 1)
dtest(uw, 0.5, 0.4, 2)
dtest(uw, 0.5, 0.4, 3)
dtest(uw, 0.5, 0.4, 4)
dtest(uw, 0.5, 0.4, 5)
dtest(uw, 0.5, 0.4, 6)

dtest(uw, 0.5, 0.499, 1)
dtest(uw, 0.5, 0.499, 2)
dtest(uw, 0.5, 0.499, 3)
dtest(uw, 0.5, 0.499, 4)
dtest(uw, 0.5, 0.499, 5)
dtest(uw, 0.5, 0.499, 6)

dtest(uw, 0., 0.4, 1)
dtest(uw, 0., 0.4, 2)
dtest(uw, 0., 0.4, 3)
dtest(uw, 0., 0.4, 4)
dtest(uw, 0., 0.4, 5)
dtest(uw, 0., 0.4, 6)

dtest(uw, 0., 0.0, 1)
dtest(uw, 0., 0.0, 2)
dtest(uw, 0., 0.0, 3)
dtest(uw, 0., 0.0, 4)
dtest(uw, 0., 0.0, 5)
# exact, mais effets pervers sur le fit (steps 0 et x₀ devraient être équivalents ?)
dtest(uw, 0., 0.0, 6)

dtest(uw, 1., 0.4, 1)
dtest(uw, 1., 0.4, 2)
dtest(uw, 1., 0.4, 3)
dtest(uw, 1., 0.4, 4)
dtest(uw, 1., 0.4, 5)
dtest(uw, 1., 0.4, 6)

dtest(uw, 1., 1.0, 1)
dtest(uw, 1., 1.0, 2)
dtest(uw, 1., 1.0, 3)
dtest(uw, 1., 1.0, 4)
dtest(uw, 1., 1.0, 5)
dtest(uw, 1., 1.0, 6)

end

end # module
