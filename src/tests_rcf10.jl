#############################################################################
#
#  testing model w/ RCF profiles
#
#############################################################################

workspace()

using VegaLite
using JLD
using BenchmarkTools

include("RNADE-2.jl")
using RNADE

include("model-bernstein-1.jl")


xs = rand(50)
pars = Pars(Bernstein(5), 10, 50)
dpars = Pars(Bernstein(5), 10, 50)
RNADE.init(pars)
@benchmark RNADE.xdloglik!(xs, pars, dpars)
# 120 μs - with type PDIST
# 189 μs - with module RNADE cleanup


score(pars::Pars, dat, f=xloglik) = mean(f(dat[:,j], pars)[1] for j in 1:size(dat,2))

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

        RNADE.zeros!(dpars)
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            RNADE.add!(dpars, df!(dat[:,yi], pars, dparsi) )
        end
        dmax = RNADE.maximum(dpars)
        (100./dmax < α*kscale/chunksize) && print("+")
        RNADE.scal!(dpars, - min(100./dmax, α*kscale/chunksize))
        # clamp!(scal!(dpars, -α*kscale/chunksize), -1., 1.)
        RNADE.add!(pars, dpars)
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
function plot_avg(train_set, parss, nbstart=10, oversample=10)
  px = repeat(linspace(0., 1., Nd), outer=1+length(parss))
  pm = zeros(Nd, length(parss) + 1)
  cn = repeat(["real"; ["simul$i" for i in 1:length(parss)]], inner=Nd)

  # subset of examples defined at least up to nbstart
  # nbstart = 30
  korrect = vec(mapslices(v -> all( x != -1. for x in v ), train_set[1:nbstart,:], 1))
  ikorrect = findin(korrect, [true])
  nbkorrect = length(ikorrect)

  showall(train_set[1:nbstart,1])
  all( x != -1. for x in train_set[1:nbstart,1] )
  collect( x != -1. for x in train_set[1:nbstart,1] )

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

# mean( diff(dat1,1) .== 0. ) # pas de mouv dans 80% des cas
# sum( diff(dat1,1), 2 )
# data_values(x=[1.:49;], y=dtir) +
#      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)



############  model
Ns = size(train_set,2)
Nh = 10
dist = Bernstein(5)

pars₀ = Pars(dist, Nh, Nd)
# RNADE.scal!(pars₀, 0.)
RNADE.init(pars₀)
score(pars₀, train_set)
score(pars₀, test_set)
parss = Pars[]
# push!(parss, deepcopy(pars₀))

pars = sgd(pars₀,
           maxtime=10, chunksize=100, maxsteps=100000,
           kscale=1., cbinterval=10, k0=1e-2)
push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=10, chunksize=100, maxsteps=100000,
           kscale=0.5, cbinterval=10, k0=1e-3)
push!(parss, deepcopy(pars))

pars = sgd(pars,
           maxtime=60, chunksize=100, maxsteps=100000,
           kscale=1., cbinterval=10, k0=1e-3)
push!(parss, deepcopy(pars))

plot_avg(train_set, parss)
save("c:/temp/pars.jld", "calc2", pars)

jldopen(file -> write(file, "calc3", pars), "c:/temp/pars2.jld", "r+")

sd = load("c:/temp/pars2.jld")

sda = collect(values(sd))
typeof(sda)
plot_avg(train_set, collect(values(sd)), 0)

plot_ex(parss[end], 3)

plot_avg2(train_set[:,1:500], parss[end:end], [0.5,0.5,0.5,0.5])
