#############################################################################
#
#  avec plus de hidden units
#
#############################################################################

using VegaLite
using JLD


const Nd = 50  # variable size to estimate
const Nh = 10  # number of hidden units
const Ne = 5   # number of mixture elements

include("setup12.jl")


score(pars::Pars, dat, f=xloglik) = mean(xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2))

function sgd(pars₀;
             f::Function = xloglik,
             df!::Function = xdloglik!,
             dat = train_set,
             datt = test_set,
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

            #  dat=train_set;
            #  maxtime=10; maxsteps=1000; chunksize=100;
            #  kscale=1e-4; cbinterval=100; k0=1e-3

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(1:size(dat,2))
    datstate = start(datiter)

    pars = deepcopy(pars₀)
    dparsi = deepcopy(pars)
    dpars = deepcopy(pars)

    # df! = xdloglik!

    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        zeros!(dpars)
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            add!(dpars, df!(dat[:,yi], pars, dparsi) )
        end
        # dpars.Vm[5]
        # sum( pars.Vm[5] .== pars₀.Vm[5] )
        add!(pars, scal!(dpars, -α*kscale))
        α = 1. / (1 + t*k0)

        any(isnan(pars.c)) && break

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll = score(pars, dat)
            llt = score(pars, datt)
            println("$t : α = $(round(α,3)), train : $(round(ll,1)), test : $(round(llt,1))")
        end
    end

    pars
end

sgd2(pars)

###########  read data

(dat0, hdr) = readcsv("c:/temp/rcf_matrix.csv", header=true)
dat0

Nd = 50  # variable size to estimate
Ns0 = Int64(size(dat0,1) / Nd)
dat1 = reshape(dat0[:,4] * 1., Nd, Ns0)
extrema(dat1)

dat1 = clamp(dat1, 0.001, 0.999)
extrema(dat1)

mean( diff(dat1,1) .== 0. ) # pas de mouv dans 80% des cas

sum( diff(dat1,1), 2 )

dtir = sum( diff(dat1[:,1:500],1), 2 )

data_values(x=[1.:49;], y=dtir) +
     mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)


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
scal!(pars₀, 0.0001)
score(pars₀, train_set)
score(pars₀, test_set)

pars = sgd(pars₀,
           maxtime=40, chunksize=100,
           kscale=1e-2, cbinterval=25, k0=5e-3)

# a,b = exp(), sigmoide : 400 : α = 0.333, train : -85.5, test : -86.1

#################### a,b = transfo, sigmoide
pars₀ = Pars(Nh, Nd, Ne)
scal!(pars₀, 10.)

pars = sgd(pars,
           maxtime=40, chunksize=30,
           kscale=1e-2, cbinterval=50, k0=5e-3)

pars = sgd(pars₀, xloglik, xdloglik!,
          maxtime=50, chunksize=100,
          kscale=1e-4, cbinterval=25, k0=5e-3)


# a,b = transfo, sigmoide : 350 : α = 0.364, train : -82.0, test : -81.5

pars = sgd(pars, xloglik, xdloglik!,
          maxtime=120, chunksize=100,
          kscale=1e-3, cbinterval=25, k0=1e-3)
score_train(pars)


pars = sgd(pars, xloglik, xdloglik!,
           maxtime=120, chunksize=100,
           kscale=1e-4, cbinterval=25, k0=1e-3)



# * 0.1 : 125 : α = 0.889, train : -151.3, test : -146.3
# *0.1, *0.1 : 125 : α = 0.889, train : -144.7, test : -140.0
# * 1 : 125 : α = 0.889, train : -156.3, test : -151.2

pars = sgd(pars, xloglik, xdloglik!,
           maxtime=120, chunksize=100,
           kscale=1e-2, cbinterval=25, k0=1e-3)

pars = sgd(pars, xloglik, xdloglik!,
           maxtime=300, chunksize=100,
           kscale=5e-2, cbinterval=25, k0=1e-3)

sum( pars.Vm[1] .== pars₀.Vm[1] )
sum( pars.Vm[5] .== pars₀.Vm[5] )

# sigmoide + a+1, b+1 modele  :1000 : α = 0.5, train : -177.3, test : -176.2


##############  drawing

spars = deepcopy(pars)
score(spars, train_set)
using JLD
save("/home/fred/spars1.jld", "spars", spars)

xs2, lls, ll = xsample([0.], spars)


#### à partir de t₀ = 0
nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples)'
py = similar(px)

for i in 1:nbexamples
  py[:,i] = xsample([rand();], spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_nominal(:cn)

########### moyenne 1
px = repeat(linspace(0., 1., Nd), outer=2)
pm = zeros(Nd,2)
cn=repeat(["real"; "simul"], inner=Nd)

nbexamples = 500
pm[:,1] = vec(mapslices(mean, train_set[:,1:nbexamples], 2)) # moyenne training set


spars = deepcopy(pars)
for j in 1:nbexamples
  pm[:,2] .+= xsample([0.5, 0.45, 0.42, 0.38], spars)[1]
end
pm[:,2] ./= nbexamples

data_values(x=vec(px), y=vec(pm), cn=cn) +
  mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
  encoding_color_nominal(:cn)

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
