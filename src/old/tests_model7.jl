#############################################################################
#
#  testing model w/ RCF profiles
#
#############################################################################

module Trying
end



module Trying

using VegaLite
# using JLD


const Nd = 50  # variable size to estimate
const Nh = 10  # number of hidden units
const Ne = 7   # number of mixture elements

include("setup15.jl")

import .RNADE: Pars, xloglik, xdloglik!, xsample, init
import .RNADE: zeros!, scal!, clamp!, add!, maximum

score(pars::Pars, dat, f=xloglik) = mean(f(dat[:,j], pars)[1] for j in 1:size(dat,2))
# score_train(pars::Pars) = score(pars, train_set)
# score_test(pars::Pars) =  score(pars, test_set)

function sgd(pars₀;
             f::Function = xloglik,
             df!::Function = xdloglik!,
             dat  = train_set,
             datt = test_set,
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(1:size(dat,2))
    datstate = start(datiter)

    pars   = deepcopy(pars₀)
    dparsi = deepcopy(pars)
    dpars  = deepcopy(pars)

    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        zeros!(dpars)
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            add!(dpars, df!(dat[:,yi], pars, dparsi) )
        end
        dmax = maximum(dpars)
        (100./dmax < α*kscale/chunksize) && print("+")
        scal!(dpars, - min(100./dmax, α*kscale/chunksize))
        # clamp!(scal!(dpars, -α*kscale/chunksize), -1., 1.)
        add!(pars, dpars)
        α = 1. / (1 + t*k0)

        if any(isnan(pars.c))
          break
        end

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll  = score(pars,  dat, f)
            llt = score(pars, datt, f)
            println("$t : α = $(round(α,3)), train : $(round(ll,1)), test : $(round(llt,1))")
        end
    end

    pars
end


#### à partir de t₀ = 0
function plot_ex{Nh,Nd,Ne}(pars::Pars{Nh,Nd,Ne}, nbexamples)
  nbexamples = 5
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
function plot_avg(train_set, parss, nbstart=10)
  px = repeat(linspace(0., 1., Nd), outer=1+length(parss))
  pm = zeros(Nd, length(parss) + 1)
  cn=repeat(["real"; ["simul$i" for i in 1:length(parss)]], inner=Nd)

  nbexamples = size(train_set,2)
  pm[:,1] = vec(mapslices(mean, train_set[:,1:nbexamples], 2)) # moyenne training set

  for (i,p) in enumerate(parss)
    for j in 1:nbexamples
      pm[:,i+1] .+= xsample(train_set[1:nbstart,j], p)[1]
    end
    pm[:,i+1] ./= nbexamples
  end

  data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)
end



###########  read data

(dat0, hdr) = readcsv("c:/temp/rcf_matrix.csv", header=true)
dat0

Nd = 50  # variable size to estimate
Ns0 = Int64(size(dat0,1) / Nd)
dat1 = reshape(dat0[:,4] * 1., Nd, Ns0)
extrema(dat1)

dat1 = clamp(dat1, 0.0001, 0.9999)
extrema(dat1)

# split train / test

ids = shuffle(1:Ns0)

train_set = dat1[:, ids[1:3500]]
test_set  = dat1[:, ids[3501:end]]

# mean( diff(dat1,1) .== 0. ) # pas de mouv dans 80% des cas
# sum( diff(dat1,1), 2 )
# data_values(x=[1.:49;], y=dtir) +
#      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)



### alternative pour test

begin
  train_set = 0.5 * ones(50,500)
  train_set .+= cumsum(rand(Normal(0.,0.01),50,500),1)
  clamp!(train_set, 0.001, 0.999)

  test_set  = 0.5 * ones(50,100)
  test_set .+= cumsum(rand(Normal(0.,0.01),50,100),1)
  clamp!(test_set, 0.001, 0.999)
end

### alternative pour test

begin
  function draw()
      if rand() > 0.5
          xs = sin.( linspace(0,pi,50) )
      else
          x0 = rand()
          xs = x0 * exp.( linspace(0,-3,50) )
      end
      xs += cumsum(rand(Normal(0.,0.01), 50))
      clamp!(xs, 0.001, 0.999)
      xs
  end

  train_set = hcat( [draw() for i in 1:500]... )

  test_set = hcat( [draw() for i in 1:100]... )

  mean(train_set)
end
######


Ns = size(train_set,2)




############  model

pars₀ = RNADE.Pars(Nh, Nd, Ne)
RNADE.scal!(pars₀, 0.00001)
RNADE.init(pars₀)
score(pars₀, train_set, xloglik)
score(pars₀, test_set, xloglik)
parss = Pars[]

# pars = sgd(pars₀, xloglik, xdloglik!,maxsteps=100,
#            maxtime=60, chunksize=100,
#            kscale=1e-3, cbinterval=25, k0=1e-2)

pars = sgd(pars₀,
           maxtime=30, chunksize=100, maxsteps=100000,
           kscale=10., cbinterval=25, k0=2e-2)
push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=120, chunksize=100, maxsteps=100000,
           kscale=0.2, cbinterval=25, k0=1e-3)
push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=120, chunksize=100, maxsteps=100000,
           kscale=0.1, cbinterval=25, k0=1e-3)
push!(parss, deepcopy(pars))


pars = sgd(pars,
          maxtime=1000, chunksize=10, maxsteps=100000,
          kscale=1., cbinterval=250, k0=1e-5)
push!(parss, deepcopy(pars))

pars = sgd(pars,
          maxtime=300, chunksize=10, maxsteps=100000,
          kscale=0.1, cbinterval=250, k0=1e-5)
push!(parss, deepcopy(pars))


# setup15 : -183.3
# setup16 : -263.7, mais moyenne en-dessous !


# 60s, 2e-1 : 450 : α = 0.526, train : -202.2, test : -202.4
# 60s, 1e-1 : 475 : α = 0.513, train : -202.2, test : -202.4
# 60s, 1e-2 : 450 : α = 0.526, train : 53.5, test : 53.3


# 60s, 2e-1, chunk=1 : 42500 : α = 0.012, train : -202.0, test : -202.2
# 60s, 1e0, chunk=1, k0=2e-5 30000 : α = 0.625, train : -162.8, test : -162.8

# Avec pars0 écrasé
# 60s, 1e0, chunk=100, k0=2e-3 : 250 : α = 0.667, train : -180.4, test : -180.5
# 60s, 1e-1, chunk=100, k0=2e-3 : 225 : α = 0.69, train : -196.9, test : -197.2

# Avec pars0 normal
# 60s, 1e-1, chunk=100, k0=2e-3 : 225 : α = 0.69, train : -200.7, test : -201.0
# 60s, 1e-1, chunk=100, k0=2e-3 : 225 : α = 0.69, train : -200.0, test : -200.2

# Avec pars0 normal + regul fac = 10.
# 60s, 1e-1, chunk=100, k0=2e-3 : 225 : α = 0.69, train : -148.1, test : -148.1

# Avec pars0 normal + regul fac = 0.
# 60s, 1e-1, chunk=100, k0=2e-3 : 200 : α = 0.714, train : -214.9, test : -215.1



plot_avg(train_set[:,1:500], parss)
plot_avg(train_set[:,1:500], parss[end:end])
plot_ex(parss[end], 5)

plot_avg2(train_set[:,1:500], parss[end:end], [0.5,0.5,0.5,0.5])



########## avec average du modèle sur plus d'échantillons
py = Float64[]
  cn = String[]

  nbexamples = size(train_set,2)

  # sdv = vec(mapslices(std, train_set[:,1:nbexamples], 2))
  mea = vec(mapslices(mean, train_set[:,1:nbexamples], 2))
  # append!(py, mea + 0.5 * sdv)
  # append!(cn, ["upper"])
  # append!(py, mea - 0.5 * sdv)
  # append!(cn, ["lower"])
  append!(py, mea)
  append!(cn, ["mean"])

  pm = zeros(Nd)
  for i in 1:10
    for j in 1:nbexamples
      pm .+= xsample(train_set[1:10,j], parss[end])[1]
    end
  end
  pm ./= 10nbexamples
  append!(py, pm)
  append!(cn, ["simul x10"])

  px = linspace(0., 1., Nd)
  data_values(x=repeat(px, outer=length(cn)), y=vec(py), col=repeat(cn, inner=Nd)) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:col)


########## avec average du modèle sur plus d'échantillons
py = Float64[]
  cn = String[]

  nbexamples = size(train_set,2)

  pm = zeros(Nd)
  for i in 1:10
    for j in 1:nbexamples
      pm .+= xsample([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.], parss[end])[1]
    end
  end
  pm ./= 10nbexamples
  append!(py, pm)
  append!(cn, ["bump puis zéro"])

  pm = zeros(Nd)
  for i in 1:10
    for j in 1:nbexamples
      pm .+= xsample([1., 1., 1., 1., 1., 1., 1., 1., 0., 0.], parss[end])[1]
    end
  end
  pm ./= 10nbexamples
  append!(py, pm)
  append!(cn, ["gros bump puis zéro"])

  pm = zeros(Nd)
  for i in 1:10
    for j in 1:nbexamples
      pm .+= xsample(zeros(10), parss[end])[1]
    end
  end
  pm ./= 10nbexamples
  append!(py, pm)
  append!(cn, ["zéro"])

  px = linspace(0., 1., Nd)
  data_values(x=repeat(px, outer=length(cn)), y=vec(py), col=repeat(cn, inner=Nd)) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:col)


##############  drawing

spars = deepcopy(pars)
score(spars, train_set)
using JLD
save("/home/fred/spars1.jld", "spars", spars)

xs2, lls, ll = xsample([0.], spars)


#### à partir de t₀ = 0
function plot_ex{Nh,Nd,Ne}(pars::Pars{Nh,Nd,Ne}, nbexamples)
  nbexamples = 5
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
function plot_avg(train_set, parss)
  px = repeat(linspace(0., 1., Nd), outer=1+length(parss))
  pm = zeros(Nd, length(parss) + 1)
  cn=repeat(["real"; ["simul$i" for i in 1:length(parss)]], inner=Nd)

  nbexamples = size(train_set,2)
  pm[:,1] = vec(mapslices(mean, train_set[:,1:nbexamples], 2)) # moyenne training set

  for (i,p) in enumerate(parss)
    for j in 1:nbexamples
      pm[:,i+1] .+= xsample(train_set[1:10,j], p)[1]
    end
    pm[:,i+1] ./= nbexamples
  end

  data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)
end




########### moyenne 2
function plot_avg2(train_set, parss, starts)
  px = repeat(linspace(0., 1., Nd), outer=length(parss))
  pm = zeros(Nd, length(parss))
  cn=repeat(["simul$i" for i in 1:length(parss)], inner=Nd)

  nbexamples = size(train_set,2)
  for (i,p) in enumerate(parss)
    for j in 1:nbexamples
      pm[:,i] .+= xsample(starts, p)[1]
    end
    pm[:,i] ./= nbexamples
  end

  data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)
end

########### moyenne 1 bis
px = repeat(linspace(0., 1., Nd), outer=2)
pm = zeros(Nd,2)
cn=repeat(["real"; "simul"], inner=Nd)

nbexamples = 500
pm[:,1] = vec(mapslices(mean, train_set[:,1:nbexamples], 2)) # moyenne training set

spars = deepcopy(pars)
for j in 1:nbexamples
    pm[:,2] .+= xsample(train_set[1:10,j], spars)[1]
end
pm[:,2] ./= nbexamples

data_values(x=vec(px), y=vec(pm), cn=cn) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
    encoding_color_nominal(:cn)

sum( spars.Vm[1] .== pars₀.Vm[1] )
sum( spars.Vm[5] .== pars₀.Vm[5] )



########### moyenne 2

pars₀ = Pars(Nh, Nd, Ne)

sum( parss[3].Vm[1] .== pars₀.Vm[1] )
sum( parss[3].Vm[5] .== pars₀.Vm[5] )

sum( parss[3].Vn[1] .== pars₀.Vn[1] )
sum( parss[3].Vn[5] .== pars₀.Vn[5] )

sum( parss[3].Vw[1] .== pars₀.Vw[1] )
sum( parss[3].Vw[5] .== pars₀.Vw[5] )

sum( parss[3].bm[1] .== pars₀.bm[1] )
sum( parss[3].bm[5] .== pars₀.bm[5] )

sum( parss[3].bn[1] .== pars₀.bn[1] )
sum( parss[3].bn[5] .== pars₀.bn[5] )

sum( parss[3].bw[1] .== pars₀.bw[1] )
sum( parss[3].bw[5] .== pars₀.bw[5] )

sum( parss[3].W .== pars₀.W )
sum( parss[3].c .== pars₀.c )


res = xdloglik!(xs, pars₀, dpars)

dpars

Np = 6
px = repeat(linspace(0., 1., Nd), outer=1+Np)
pm = zeros(Nd,Np)
cn=repeat(["real"; collect(1:Np)], inner=Nd)

nbexamples = 500
pmref = vec(mapslices(mean, train_set[:,1:nbexamples], 2)) # moyenne training set

parss = Array(Pars, Np)
parss[1] = pars₀
for i in 2:Np
  parss[i] = sgd(parss[i-1], xloglik, xdloglik!,
            maxtime=10, chunksize=100,
            kscale=1e-2, cbinterval=25, k0=5e-3)

  for j in 1:nbexamples
    pm[:,i] .+= xsample([train_set[1,j]], parss[i])[1]
  end
  pm[:,i] ./= nbexamples
end

for i in 2:Np
  for j in 1:nbexamples
    pm[:,i] .+= xsample(train_set[1:20,j], parss[i])[1]
  end
  pm[:,i] ./= nbexamples
end

# nbexamples = Ns
# for i in 1:nbexamples
#   # pm .+= xsample([train_set[2,i]], spars)[1]
#   pm .+= xsample([0.], pars₀)[1]
# end
# pm ./= nbexamples

data_values(x=vec(px), y=vec([pmref pm]), cn=cn) +
  mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
  encoding_color_nominal(:cn)

#### profils du training set
nbexamples = 10
si = sample(1:Ns,nbexamples)
px = linspace(0., 1., Nd) * ones(nbexamples)'
py = similar(px)

for (i,idx) in enumerate(si)
  py[:,i] = dat[:,idx]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)


end  # of module
