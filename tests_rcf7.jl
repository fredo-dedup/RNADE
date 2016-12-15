#############################################################################
#
#  testing model w/ RCF profiles
#
#############################################################################

using VegaLite
using JLD


Nd = 50  # variable size to estimate
Nh = 10  # number of hidden units
Ne = 5   # number of mixture elements

include("setup11.jl")

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
        clamp!(scal!(dpars, -α*kscale), -1., 1.)
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


###########  read data

(dat0, hdr) = readcsv("c:/temp/rcf_matrix.csv", header=true)
dat0

Nd = 50  # variable size to estimate
Ns0 = Int64(size(dat0,1) / Nd)
dat1 = reshape(dat0[:,4] * 1., Nd, Ns0)
extrema(dat1)

dat1 = clamp(dat1, 0.0001, 0.9999)
extrema(dat1)

mean( diff(dat1,1) .== 0. ) # pas de mouv dans 80% des cas

sum( diff(dat1,1), 2 )

# data_values(x=[1.:49;], y=dtir) +
#      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)


# split train / test

ids = shuffle(1:Ns0)

train_set = dat1[:, ids[1:3500]]
test_set  = dat1[:, ids[3501:end]]

### alternative pour test

train_set = 0.5 * ones(50,500)
train_set .+= cumsum(rand(Normal(0.,0.01),50,500),1)
clamp!(train_set, 0.001, 0.999)

test_set  = 0.5 * ones(50,100)
test_set .+= cumsum(rand(Normal(0.,0.01),50,100),1)
clamp!(test_set, 0.001, 0.999)

### alternative pour test

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

######


Ns = size(train_set,2)




############  model

pars₀ = Pars(Nh, Nd, Ne)
scal!(pars₀, 0.1)
score(pars₀, train_set, xloglik)
score(pars₀, test_set, xloglik)
parss = Pars[]

# pars = sgd(pars₀, xloglik, xdloglik!,maxsteps=100,
#            maxtime=60, chunksize=100,
#            kscale=1e-3, cbinterval=25, k0=1e-2)

pars = sgd(pars₀,
           maxtime=300, chunksize=100, maxsteps=100000,
           kscale=1e-1, cbinterval=25, k0=2e-3)


score(pars, train_set, xloglik)
score(pars, test_set, xloglik)

push!(parss, deepcopy(pars))


# a,b = exp(), sigmoide : 400 : α = 0.333, train : -85.5, test : -86.1
# a,b = exp(), sigmoide + points 0,x,1 : 500 : α = 0.286, train : -273.6, test : -266.1

plot_avg(train_set[:,1:500], parss[end:end])
plot_ex(parss[end], 5)

plot_avg2(train_set[:,1:500], parss[end:end], [0.5,0.5,0.5,0.5])


pars = sgd(pars,
           maxtime=120, chunksize=100,
           kscale=1e-1, cbinterval=25, k0=5e-3)

push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=300, chunksize=100,
           kscale=1e-5, cbinterval=25, k0=5e-3)

push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=60, chunksize=100,
           kscale=1e-3, cbinterval=25, k0=5e-3)

push!(parss, deepcopy(pars))


##############  drawing

spars = deepcopy(pars)
score_train(spars)
using JLD
save("/home/fred/spars1.jld", "spars", spars)

xs2, lls, ll = xsample([0.], spars)


#### à partir de t₀ = 0
function plot_ex(pars, nbexamples)
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
