using ReverseDiffSource

# for some reason type inference does not work for ^(Float64, Float64)
import Base.^
^(a::Float64,b::Float64)::Float64 = exp(log(a)*b)

3. ^ 2.


ek3 = quote
    em = exp(m)
    a = sqrt(em * exp(n))
    b = a / em
    a*b*x^(a - 1.) * (1. - x^a)^(b - 1.)
end

# i = 1
# fk3(p, pars[i,1], pars[i,2])
# m = pars[i,1]
# n = pars[i,2]
# x = 0.945

@eval fk3(x::Float64,m::Float64,n::Float64) = $ek3
edk3 = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x, allorders=false)
@eval fdk3(x::Float64,m::Float64,n::Float64) = $edk3
edk3a = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x)
@eval fdk3a(x::Float64,m::Float64,n::Float64) = $edk3a

# pars = cpars
# p = 0.945
function loglik(pars::Matrix{Float64}, p::Float64)
    ws = exp.(pars[:,3])
    ws ./= sum(ws)
    pm = dot(ws, fk3.(p, pars[:,1], pars[:,2]))
    -log(pm)
end
# loglik(pars, 0.5)
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


function dloglik!(pars::Matrix{Float64}, dpars::Matrix{Float64}, p::Float64)
    # p = 0.5
    ws0 = exp.(pars[:,3])
    sws0 = sum(ws0)
    ws = ws0 ./ sws0
    ps = fk3.(p, pars[:,1], pars[:,2])
    pm = dot(ws, ps)

    _tmp5 = - ps ./ pm
    dpars[:,3] = ws0 ./ sws0 .* ( _tmp5 + ( sum(-ws0 .* _tmp5) / sws0 ) )

    dv0 = -1/pm
    for i in 1:size(pars,1)
        dpars[i,1], dpars[i,2] = fdk3(p, pars[i,1], pars[i,2])
        dpars[i,1] *= ws[i] * dv0
        dpars[i,2] *= ws[i] * dv0
    end
    dpars
end


function fcdf3(m::Float64,n::Float64,x::Float64)
  em = exp(m)
  a = sqrt(em * exp(n))
  b = a / em
  1. - (1 - x^a)^b
end


# fcdf3(-5.,10.,0.5)
# fcdf3.(-2.,2.,[0:0.1:1;])

function ficdf3(m::Float64,n::Float64,p::Float64)
  em = exp(m)
  a = sqrt(em * exp(n))
  b = a / em
  (1. - (1. - p)^(1/b))^(1/a)
end

# fcdf3(0.,2.,0.5)
# ficdf3(0.,2.,0.361)
# ficdf3(0.,2.,0.5)
