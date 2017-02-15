#############################################################################
#
#  testing model w/ RCF profiles
#
#############################################################################

module Try
end


module Try

using VegaLite
using JLD
using BenchmarkTools

include("RNADE-3.jl")
using .RNADE

include("model-bernstein-5.jl")


xs = rand(50)
pars = Pars(Bernstein5(5), 10, 50)
dpars = Pars(Bernstein5(5), 10, 50)
init(pars)
@benchmark xdloglik!(xs, pars, dpars)
# 120 μs - with type PDIST
# 189 μs - with module RNADE cleanup


#### à partir de t₀ = 0
function plot_ex{Nh,Nd}(pars::Pars{Nh,Nd}, nbexamples)
  # nbexamples = 5
  px = linspace(0., 1., Nd) * ones(nbexamples)'
  py = similar(px)

  for i in 1:nbexamples
    py[:,i] = xsample([0., 0., 0.], pars)[1]
  end

  data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples;], inner=[Nd])) +
        mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
        encoding_color_nominal(:cn)
end

########### moyenne 1
function plot_avg(train_set, parss; nbstart=10, oversample=10)
  # nbstart = 10
  px = repeat(linspace(0., 1., Nd), outer=1+length(parss))
  pm = zeros(Nd, length(parss) + 1)
  cn = repeat(["real"; ["simul$i" for i in 1:length(parss)]], inner=Nd)

  # subset of examples defined at least up to nbstart
  # nbstart = 30
  korrect = vec(mapslices(v -> all( x != -1. for x in v ), train_set[1:nbstart,:], 1))
  ikorrect = findin(korrect, [true])
  nbkorrect = length(ikorrect)

  # showall(train_set[1:nbstart,1])
  # extrema(train_set[1:nbstart, ikorrect])
  # mean(train_set[1:nbstart, ikorrect],2)
  # all( x != -1. for x in train_set[1:nbstart,ikorrect] )


  # mean drawn over subset
  pm[:,1] = [ mean( x for x in train_set[i,korrect] if x != -1. ) for i in 1:size(train_set,1) ] # moyenne training set

  # drawings from distributions
  for (i,p) in enumerate(parss) # i, p = 1, parss[2]
    for j in ikorrect  # j = 1
      for k in 1:oversample  # j = 3
        pm[:,i+1] .+= xsample(train_set[1:nbstart,j], p)[1]
      end
    end
    pm[:,i+1] ./= nbkorrect * oversample
  end

  data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)
end



###########  read data

dat0 = readdlm("c:/temp/rcf_matrix2.csv", header=false)
dat1 = Float64[ x == "NA" ? -1. : x for x in dat0 ]
# save("c:/temp/rcf_matrix2.jld", "dat1", dat1)

# split train / test
Nd = size(dat1,1)
Ns0 = size(dat1,2)
ids = shuffle(1:Ns0)

train_set = dat1[:, ids[1:3500]]
test_set  = dat1[:, ids[3501:end]]

# extrema(train_set)
# mean( diff(dat1,1) .== 0. ) # pas de mouv dans 80% des cas
# poss = findin(dat1, [-1.])
# sum(dat1[1,:] .== -1.)
# sum(dat1[poss+1] .!= -1.)
# sum(dat1[end,:] .== -1.)
# sum( diff(dat1,1), 2 )
# data_values(x=[1.:49;], y=dtir) +
#      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)


###########  alternate data

begin
  Nd = 50

  function draw()
      if rand() > -0.5
          xs = 0.1 + 0.5 * sin.( linspace(0,pi,Nd) )
      else
          x0 = rand()
          xs = x0 * exp.( linspace(0,-3,Nd) )
      end
      xs += cumsum(rand(Normal(0.,0.01), Nd))
      cens = rand(1:100)
      clamp!(xs, 0., 1.)
      cens < Nd && ( xs[cens+1:Nd] = -1. )
      xs
  end

  draw()

  train_set = hcat( [draw() for i in 1:500]... )
  test_set = hcat( [draw() for i in 1:100]... )

  mean(train_set)
  mean( train_set .== -1. )
  extrema(train_set)
end


############  model
Ns = size(train_set,2)
Nh = 2
dist = Bernstein5(10)

pars₀ = Pars(dist, Nh, Nd)
init(pars₀)
# RNADE.scal!(pars₀, 0.1)
score(pars₀, train_set)
score(pars₀, test_set)
parss = Pars[]
# push!(parss, deepcopy(pars₀))

pars = sgd(pars₀, train_set, test_set,
           maxtime=10, chunksize=100, maxsteps=100000,
           kscale=1., cbinterval=10, k0=1e-3)
push!(parss, deepcopy(pars))

pars = sgd(pars, train_set, test_set,
           maxtime=10, chunksize=100, maxsteps=100000,
           kscale=0.5, cbinterval=10, k0=1e-3)
push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=60, chunksize=100, maxsteps=100000,
           kscale=1., cbinterval=10, k0=1e-3)
push!(parss, deepcopy(pars))

plot_avg(train_set, parss, nbstart=10)
# save("c:/temp/pars.jld", "calc2", pars)

# sur 1 exemple
plot_avg(ones(50,1) * 0.1, parss, nbstart=1, oversample=1000)

whos(r"pars")
Base.summarysize(pars.W)


jldopen(file -> write(file, "calc3", pars), "c:/temp/pars2.jld", "r+")

sd = load("c:/temp/pars2.jld")

plot_avg(train_set, collect(values(sd)), 0)

plot_ex(parss[end], 3)

plot_avg2(train_set[:,1:500], parss[end:end], [0.5,0.5,0.5,0.5])



############################################################

###### distrib à t = 0. ######

Ns = size(train_set,2)
Nh = 10
dist = Bernstein(15)

pars₀ = Pars(dist, Nh, Nd)
RNADE.init(pars₀)
parss = Pars[]

train_set2 = copy(train_set)
train_set2[2,:] = -1.

pars = sgd(pars₀, train_set2, test_set,
           maxtime=10, chunksize=100, maxsteps=100000,
           kscale=1., cbinterval=10, k0=1e-2)
push!(parss, deepcopy(pars))
pars.c

Ne = 20
β = rand(Normal(), Ne)
dist = Bernstein5(Ne)
update!(dist, β, 0.)

β = zeros(β); dβ = zeros(β)
dβ *= 0.8
  dβ1 = similar(β)
  for xs in train_set[1,:]
    dlogpdf!(dist, xs, dβ1)
    dβ += 0.2 * dβ1
  end
  β -= 0.001 * dβ

  update!(dist, β, 0.)
  (mean( logpdf(dist, xs) for xs in train_set[1,:] ) ,
   mean( rand(dist) for i in 1:10000 ),
   mean( train_set[1,:] ) )

using StatsBase
hcat(fit(Histogram, train_set[1,:], 0:0.1:1.1, closed=:left).weights,
     fit(Histogram, collect(rand(dist) for i in 1:Ns), 0:0.1:1.1, closed=:left).weights )

######## fit sur target unique

mf(dist) = quadgk(x -> x*fp(x, dist.w, dist.binom), 0., 1.)[1]

target = 0.3
dist = Bernstein5(20)
# β = [1. ; zeros(19)]
β = rand(Normal(0.,0.1),20)
update!(dist, β, 0.)
(logpdf(dist, target), mf(dist) )

vals = Vector()
dβ = zeros(β) ; dβ₁ = zeros(β)
α = 0.8
dlogpdf!(dist, target, dβ₁)
  dβ = dβ * α + dβ₁ * (1. - α)
  β -= 10. * dβ
  # β /= sum(abs(β))
  update!(dist, β, 0.)
  push!(vals, copy(dist.w))
  (logpdf(dist, target), mf(dist), target)

vals

# mean( x*exp(-logpdf(dist, x)) for x in 0:0.001:1 )
# [ exp(logpdf(dist, x)) for x in 0:0.01:1 ]
# dlogpdf!(dist, target, dβ)
# logpdf(dist, 0.9)
# fp(0.9, dist.w, dist.binom)
# mean( x*fp(x, dist.w, dist.binom) for x in 0:0.001:1 )
# [ mean( x*fp(x, eye(5)[:,i], dist.binom) for x in 0:0.001:1 ) for i in 1:5 ]
#
# duw = similar(β)
# dfp!(0.23, [1.,0.,0.,0.,0.], dist.binom, duw) ; duw
# dfp!(0.23, [0.,1.,0.,0.,0.], dist.binom, duw) ; duw
# dfp!(0.2, [0.,0.,1.,0.,0.], dist.binom, duw) ; duw
#
# fp(0.23, [1.,0.,0.,0.,0.], dist.binom)
# fp(0.23, [0.,1.,0.,0.,0.], dist.binom)
#
# β = [0.,1.,0.,0.,0.] ; update!(dist, β, 0.)
# dlogpdf!(dist, target, dβ)

nc = length(vals[1])
  np = length(vals)
  px = repeat(1:np, outer=nc)
  cn = repeat(["v$i" for i in 1:nc], inner=np)
  pm = Float64[ vals[i][j] for i in 1:np, j in 1:nc  ]
  data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)



###########


end # of module Try
