################################################################
#  setup 7
#  retour sur pur kuma (test)
#  activation sigmoide
#  param plus linéaire
################################################################

using Distributions
using BenchmarkTools
using VegaLite
using ReverseDiffSource

const Ne = 5

@time for i in 1:1e6 ; 3.5 ^ 1.123 ; end  # 0.106s
@time for i in 1:1e6 ; exp(log(3.5) * 1.123) ; end # 0.05 plus rapide !!!

############## internal model specs  #############################

# type inference does not work ^ => creation of a power function

@inline power(a::Float64,b::Float64) = exp(log(a)*b)
@deriv_rule power(x::Real, y::Real)  x     y * power(x,y-1.) * ds
@deriv_rule power(x::Real, y::Real)  y     log(x) * power(x,y) * ds

ek3 = quote
    a = log(1+exp(m))
    b = log(1+exp(n))
    lx = log(x)
    a * b * exp(lx * (a-1.)) * exp( log( 1. - exp(lx * a) ) * (b-1.) )
    # a * b * power(x,a-1.)*power(1. - power(x,a), b-1.)
end

# TODO : optim du code généré

@eval @inline fk3(x::Float64,m::Float64,n::Float64) = $ek3

edk3 = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x, allorders=false)
@eval @inline fdk3(x::Float64,m::Float64,n::Float64) = $edk3
# edk3a = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x)
# @eval fdk3a(x::Float64,m::Float64,n::Float64) = $edk3a

@benchmark fk3(0.5, 2., 3.) # 184ns
@benchmark fdk3(0.5, 2., 3.) # 206ns

function fcdf3(m::Float64,n::Float64,x::Float64)
    a = log(1+exp(m))
    b = log(1+exp(n))
  1. - (1 - x^a)^b
end

# fcdf3(-5.,10.,0.5)
# fcdf3.(-2.,2.,[0:0.1:1;])

function ficdf3(m::Float64,n::Float64,p::Float64)
  a = log(1+exp(m))
  b = log(1+exp(n))
  (1. - (1. - p)^(1/b))^(1/a)
end

# fcdf3(0.,2.,0.5)
# ficdf3(0.,2.,0.361)
# ficdf3(0.,2.,0.5)


####### loglik definitions  #######################################

let ws = Array(Float64, Ne)
    global loglik
    @inbounds function loglik(cpars::Matrix{Float64}, x::Float64, x₀::Float64)
        @assert Ne == size(cpars,1)

        sws = 0.
        pm = 0.
        for i in 1:Ne-3
            ew = exp(cpars[i,3])
            sws += ew
            pm += ew * fk3(x, cpars[i,1], cpars[i,2])
        end

        ew = exp(cpars[Ne-2,3])
        sws += ew
        x < 0.001 && (pm += ew*1000.)

        ew = exp(cpars[Ne-1,3])
        sws += ew
        x00 = clamp(x₀-5e-4, 0.   , 0.999)
        x01 = clamp(x₀+5e-4, 0.001, 1.   )
        x00 < x < x01 && (pm += ew*1000.)

        ew = exp(cpars[Ne,3])
        sws += ew
        x>0.999 && (pm += ew*1000.)

        log(sws) - log(pm)
    end
end
# loglik(pars, 0.5, 0.5)
# exp(-loglik(pars, 0.1))
#
cpars = rand(Normal(),Ne,3)
@benchmark loglik(cpars, 0.5, 0.2) # 470ns

let ps  = Array(Float64,Ne)

    global dloglik!

    function dloglik!(cpars::Matrix{Float64}, dcpars::Matrix{Float64},
                      x::Float64, x₀::Float64)
        @assert size(cpars,1) == Ne # to avoid ugly errors

        sws = 0.
        pm = 0.
        for i in 1:Ne-3
            ew = exp(cpars[i,3])
            sws += ew
            m, n = cpars[i,1], cpars[i,2]
            psi = fk3(x, m, n)
            pm += ew * psi
            ps[i] = psi
            dcpars[i,3] = ew
            dm, dn = fdk3(x, m, n)
            dcpars[i,1] = ew * dm
            dcpars[i,2] = ew * dn
        end

        ew = exp(cpars[Ne-2,3])
        sws += ew
        if x < 0.001
            pm += ew*1000.
            ps[Ne-2] = 1000.
        else
            ps[Ne-2] = 0.
        end
        dcpars[Ne-2,3] = ew
        dcpars[Ne-2,1],dcpars[Ne-2,2] = 0., 0.

        ew = exp(cpars[Ne-1,3])
        sws += ew
        x00 = clamp(x₀-5e-4, 0.   , 0.999)
        x01 = clamp(x₀+5e-4, 0.001, 1.   )
        if x00 < x < x01
            pm += ew*1000.
            ps[Ne-1] = 1000.
        else
            ps[Ne-1] = 0.
        end
        dcpars[Ne-1,3] = ew
        dcpars[Ne-1,1],dcpars[Ne-1,2] = 0., 0.

        ew = exp(cpars[Ne,3])
        sws += ew
        if x>0.999
            pm += ew*1000.
            ps[Ne] = 1000.
        else
            ps[Ne] = 0.
        end
        dcpars[Ne,3] = ew
        dcpars[Ne,1],dcpars[Ne,2] = 0., 0.

        dv0 = -1/pm
        for i in 1:Ne
            dcpars[i,1] *= dv0
            dcpars[i,2] *= dv0
            dcpars[i,3] = (1.0 / sws - ps[i] / pm) * dcpars[i,3]
        end

        dcpars












    end
end

dcpars = zeros(cpars)
@benchmark dloglik!(cpars, dcpars, 0.2, 0.6) # 796ns

function dtest(cpars, x, x₀, indexes)
  # field, indexes = :Vm, [2, [1,1]]
  δ = 1e-8

  p0 = getindex(cpars, indexes...)
  cpars2 = deepcopy(cpars)
  setindex!(cpars2, p0+δ, indexes...)

  ed = (loglik(cpars2, x, x₀) - loglik(cpars, x, x₀)) / δ

  dcpars = zeros(cpars)
  dloglik!(cpars, dcpars, x, x₀)
  ed0 = getindex(dcpars, indexes...)
  ( ed0, ed )
end

if false
cpars = rand(5,3)

dtest(cpars, 0.5, 0.4, [1,1])
dtest(cpars, 0.5, 0.4, [2,2])

dtest(cpars, 0.5, 0.4, [5,1])
dtest(cpars, 0.5, 0.4, [5,2])
dtest(cpars, 0.5, 0.4, [3,1])
dtest(cpars, 0.5, 0.4, [4,2])

dtest(cpars, 0.5, 0.4, [1,3])
dtest(cpars, 0.5, 0.4, [5,3])
dtest(cpars, 0.5, 0.4, [5,3])

dtest(cpars, 0.5, 0.5, [1,1])
dtest(cpars, 0.5, 0.5, [2,2])

dtest(cpars, 0.5, 0.5, [1,3])
dtest(cpars, 0.5, 0.5, [3,3])
dtest(cpars, 0.5, 0.5, [4,3])
dtest(cpars, 0.5, 0.5, [5,3])

dtest(cpars, 0.0001, 0.5, [1,1])
dtest(cpars, 0.0001, 0.5, [2,2])

dtest(cpars, 0.0001, 0.5, [1,3])
dtest(cpars, 0.0001, 0.5, [3,3])
dtest(cpars, 0.0001, 0.5, [4,3])
dtest(cpars, 0.0001, 0.5, [5,3])

end

############## RNADE param type definition ####################

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



pars = Pars(Nh, Nd, Ne)
dpars = deepcopy(pars)
# whos(r"pars") # ~400kb

xs = rand(20)

############  RNADE defs #######################################

sigm(x::Float64) = 1 ./ (1+exp(-x))

# xs = test_set[:,225]

function xloglik(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
  xt = Array(Float64, nx)
  h  = Array(Float64, Nh, nx)
  cpars = Array(Float64, Ne, 3)
  for i in 1:nx # i = 33
    h[:,i] .= sigm.(a)
    # h[:,i] .= max.(0., a)

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
    # δa += δh[:,i] .* (h[:,i] .> 0.)
  end
  copy!(dpars.c, δa)
  dpars
end

xloglik(xs, pars)
xdloglik!(xs, pars, dpars)
dpars.c[1]
dpars.c

# profiling #################

Profile.clear()
Base.@profile collect(xdloglik!(train_set[:,i], pars, dpars) for i in 50:500)
Profile.print()
#
@time collect(xdloglik!(train_set[:,i], pars, dpars) for i in 50:500)
# 3.83 s
# 3.14 s
# 3.00 s
# 2.86 s
# 1.81 s
# 0.71 s
# 0.64 s

###############  testing ########################

if false
pars = scal!(Pars(Nh, Nd, Ne), 0.1)
dpars = deepcopy(pars)
xs = rand(20)

xs = train_set[:,400]

dpars = xdloglik!(xs, pars, dpars)
v0 = xloglik(xs,pars)[1]

function dtest(v0, pars, dpars, field, indexes...)
  # field, indexes = :Vm, [2, [1,1]]
  δ = 1e-8

  p0 = foldl((x,idx) -> getindex(x, idx...), getfield(pars, field), indexes)

  pars2 = deepcopy(pars)
  np = foldl((x,idx) -> getindex(x, idx...), getfield(pars2, field), indexes[1:end-1])
  setindex!(np, p0+δ, indexes[end]...)

  ed = (xloglik(xs,pars2)[1]-v0) / δ

  ed0 = foldl((x,idx) -> getindex(x, idx...), getfield(dpars, field), indexes)
  ( ed0, ed )
end


dtest(v0, pars, dpars, :Vm, 2, [1,1])
dtest(v0, pars, dpars, :Vm, 2, [10,1])
dtest(v0, pars, dpars, :Vm, 2, [1,10])
dtest(v0, pars, dpars, :Vm, 2, [10,30])
dtest(v0, pars, dpars, :Vm, 2, [10,15])

dtest(v0, pars, dpars, :Vm, 5, [1,1])
dtest(v0, pars, dpars, :Vm, 5, [10,1])
dtest(v0, pars, dpars, :Vm, 5, [1,10])
dtest(v0, pars, dpars, :Vm, 5, [10,30])
dtest(v0, pars, dpars, :Vm, 5, [10,15])

dtest(v0, pars, dpars, :Vn, 2, [1,1])
dtest(v0, pars, dpars, :Vn, 2, [10,1])
dtest(v0, pars, dpars, :Vn, 2, [1,10])
dtest(v0, pars, dpars, :Vn, 2, [10,30])
dtest(v0, pars, dpars, :Vn, 2, [10,15])

dtest(v0, pars, dpars, :Vn, 5, [1,1])
dtest(v0, pars, dpars, :Vn, 5, [10,1])
dtest(v0, pars, dpars, :Vn, 5, [1,10])
dtest(v0, pars, dpars, :Vn, 5, [10,30])
dtest(v0, pars, dpars, :Vn, 5, [10,15])

dtest(v0, pars, dpars, :Vw, 2, [1,1])
dtest(v0, pars, dpars, :Vw, 2, [10,1])
dtest(v0, pars, dpars, :Vw, 2, [1,10])
dtest(v0, pars, dpars, :Vw, 2, [10,30])
dtest(v0, pars, dpars, :Vw, 2, [10,15])

dtest(v0, pars, dpars, :Vw, 5, [1,1])
dtest(v0, pars, dpars, :Vw, 5, [10,1])
dtest(v0, pars, dpars, :Vw, 5, [1,10])
dtest(v0, pars, dpars, :Vw, 5, [10,30])
dtest(v0, pars, dpars, :Vw, 5, [10,15])



dtest(v0, pars, dpars, :bm, 2, 1)
dtest(v0, pars, dpars, :bm, 2, 10)
dtest(v0, pars, dpars, :bm, 2, 15)
dtest(v0, pars, dpars, :bm, 2, 20)
dtest(v0, pars, dpars, :bm, 2, 45)

dtest(v0, pars, dpars, :bm, 5, 1)
dtest(v0, pars, dpars, :bm, 5, 10)
dtest(v0, pars, dpars, :bm, 5, 15)
dtest(v0, pars, dpars, :bm, 5, 20)
dtest(v0, pars, dpars, :bm, 5, 45)

dtest(v0, pars, dpars, :bn, 2, 1)
dtest(v0, pars, dpars, :bn, 2, 10)
dtest(v0, pars, dpars, :bn, 2, 15)
dtest(v0, pars, dpars, :bn, 2, 20)
dtest(v0, pars, dpars, :bn, 2, 45)

dtest(v0, pars, dpars, :bn, 5, 1)
dtest(v0, pars, dpars, :bn, 5, 10)
dtest(v0, pars, dpars, :bn, 5, 15)
dtest(v0, pars, dpars, :bn, 5, 20)
dtest(v0, pars, dpars, :bn, 5, 45)

dtest(v0, pars, dpars, :bw, 2, 1)
dtest(v0, pars, dpars, :bw, 2, 10)
dtest(v0, pars, dpars, :bw, 2, 15)
dtest(v0, pars, dpars, :bw, 2, 20)
dtest(v0, pars, dpars, :bw, 2, 45)

dtest(v0, pars, dpars, :bw, 5, 1)
dtest(v0, pars, dpars, :bw, 5, 10)
dtest(v0, pars, dpars, :bw, 5, 15)
dtest(v0, pars, dpars, :bw, 5, 20)
dtest(v0, pars, dpars, :bw, 5, 45)


dtest(v0, pars, dpars, :W, [1,1])
dtest(v0, pars, dpars, :W, [10,1])
dtest(v0, pars, dpars, :W, [1,10])
dtest(v0, pars, dpars, :W, [10,30])
dtest(v0, pars, dpars, :W, [10,15])

dtest(v0, pars, dpars, :c, 1)
dtest(v0, pars, dpars, :c, 10)
dtest(v0, pars, dpars, :c, 5)

end

##########################################################################
## make a sample following the partial values given
##########################################################################

function xsample(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll = 0.
  xt = Array(Float64, Nd)
  h  = Array(Float64, Nh, Nd)
  cpars = Array(Float64, Ne, 3)

  # known part
  for i in 1:nx # i = 1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    xt[i] = loglik(cpars, xs[i], (i==1) ? xs[1] : xs[i-1])
    ll += xt[i]
    a .+= pars.W[:,i] * xs[i]
  end

  xs2 = zeros(Nd)
  xs2[1:nx] = xs
  # sampled part
  for i in nx+1:Nd # i = nx+1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      cpars[j,1] = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      cpars[j,2] = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      cpars[j,3] = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
    end

    # pick component
    wn = exp.(cpars[:,3])
    wn ./= sum(wn)
    ci = rand(Categorical(wn))

    # pick x value
    if ci == Ne   # normal component, centered on x[i-1]
      # xs2[i] = clamp(rand(Normal(xs2[i-1], 0.01)), 0.001, 0.999)
      # xs2[i] = clamp( xs2[i-1], 0.001, 0.999)
      xs2[i] = ficdf3(cpars[ci,1], cpars[ci,2], rand())
    else
      xs2[i] = ficdf3(cpars[ci,1], cpars[ci,2], rand())
    end

    xt[i] = loglik(cpars, xs2[i], clamp(xs2[i-1],0.001,0.999))
    ll += xt[i]
    a  .+= pars.W[:,i] * xs2[i]
  end


  xs2, xt, ll
end
