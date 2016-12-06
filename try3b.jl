## Mixture of Kumaraswamy distributions  ####
using Distributions
using BenchmarkTools

Ne = 5   # 5 components

# 3 params m,n and weigth

pars = rand(Normal(),Ne,3)

function loglik(pars::Matrix{Float64}, p::Float64)
    ws = exp.(pars[:,3])
    ws ./= sum(ws)
    pm = dot(ws, fk3.(p, pars[:,1], pars[:,2]))
    -log(pm)
end
loglik(pars, 0.5)
exp(-loglik(pars, 0.1))

@benchmark loglik(pars, 0.5) # 28us


delta = 1e-4
pars1 = copy(pars) ; pars1[1,1] += delta
(loglik(pars1,0.9)-loglik(pars,0.9))/delta

pars1 = copy(pars) ; pars1[1,2] += delta
(loglik(pars1,0.9)-loglik(pars,0.9))/delta

pars1 = copy(pars) ; pars1[1,3] += delta
(loglik(pars1,0.9)-loglik(pars,0.9))/delta

dloglik(pars, 0.9)[6,:]


function dloglik(pars::Matrix{Float64}, p::Float64)
    # p = 0.5
    ws0 = exp.(pars[:,3])
    sws0 = sum(ws0)
    ws = ws0 ./ sws0
    ps = fk3.(p, pars[:,1], pars[:,2])
    pm = dot(ws, ps)

    dps = similar(pars)

    _tmp5 = - ps ./ pm
    dps[:,3] = ws0 ./ sws0 .* ( _tmp5 + ( sum(-ws0 .* _tmp5) / sws0 ) )

    dv0 = -1/pm
    for i in 1:size(pars,1)
        dps[i,1], dps[i,2] = fdk3(p, pars[i,1], pars[i,2])
        dps[i,1] *= ws[i] * dv0
        dps[i,2] *= ws[i] * dv0
    end
    dps
end

@benchmark dloglik(pars, 0.5) # 33us




















deriv(pars, 0.5)

641/503
641/480


0.0641 / ws0[i]

0.0641 / ws0[i] - 1 / sum(ws0)

ws[i] * fk3.(p, pars[i,1], pars[i,2])
sum(ws0)


0.0641 - 1/sum(ws0)




_tmp1 = exp(x)
_tmp2 = sum(ws0[i] + b)
_tmp3 = - 1.0 / ws[i]
ws0[i] * (  - sum(ws0) + ws0[i] ) / (ws[i] * sum(ws0) * sum(ws0))

_tmp1 = exp(x)
_tmp2 = sum(_tmp1 + b)
_tmp3 = -1.0 / (_tmp1 / _tmp2) = - _tmp2 / _tmp1
_tmp1 * _tmp3 ( 1 / _tmp2  - _tmp1 / (_tmp2 * _tmp2) )

 sum(ws0) ( 1 / sum(ws0)  - ws0[i] / (sum(ws0) * sum(ws0)) )
1 - ws0[i] / sum(ws0)

a = ones(Ne)
tt = quote
    w = exp(x)
    w1 = w / sum(w)
    p = dot(w1,a)
    -log(p)
end

show(rdiff(tt, x=Vector{Float64}, a =Float64, ignore=[:a], allorders=false))
show(rdiff(tt, x=Vector{Float64}, allorders=false))

dtt = rdiff(tt, x =Vector{Float64}, a =Float64, ignore=[:a], allorders=false)

@eval ( x = pars[:,3] ; a = fk3(p, pars[i,1], pars[i,2]) ; $dtt )


_tmp1 = exp(x)
_tmp2 = sum(exp(x))
_tmp3 = size(exp(x))

ws = exp(x) / sum(exp(x))
_tmp5 = - a ./ dot(ws,a)

exp(x) .* ( _tmp5 ./ sum(exp(x)) + ones(_tmp3) .* (  sum(-exp(x) .* _tmp5) / (sum(exp(x)) * sum(exp(x)))) )

exp(x) .* ( _tmp5 ./ sum(exp(x))) .* ( 1. + ones(_tmp3) .* (  sum(-exp(x) .* _tmp5) / (sum(exp(x)) * sum(exp(x)))) )

ws0[i] * ( -fk3(p, pars[i,1], pars[i,2]) / pm / sum(ws0))


sws0 = sum(ws0)

_tmp5 = - fk3.(p, pars[:,1], pars[:,2]) ./ pm

ws0 ./ sws0 .* ( _tmp5 + ones(Ne) .* ( sum(-ws0 .* _tmp5) / sws0 ) )
