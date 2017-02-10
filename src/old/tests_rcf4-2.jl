#############################################################################
#
#  avec plus de hidden units
#
#############################################################################


using VegaLite

function sgd(pars₀, f, df!, dat;
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(1:size(dat,2))
    datstate = start(datiter)

    pars = deepcopy(pars₀)
    dpars = deepcopy(pars) ; zeros!(dpars)

    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        zeros!(dpars)
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            add!(dpars, df!(dat[:,yi], pars, dpars) )
        end
        add!(pars, scal!(dpars, -α*kscale))
        α = 1. / (1 + t*k0)

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll = mean( f(dat[:,j], pars)[1] for j in 1:Ns )
            println("$t : α = $(round(α,3)) : $(round(ll,0))")
        end
    end

    pars
end


###########  read data

(dat0, hdr) = readcsv("c:/temp/rcf_matrix.csv", header=true)
dat0

Nd = 50  # variable size to estimate
Ns = Int64(size(dat0,1) / Nd)
dat = reshape(dat0[:,4] * 1., Nd, Ns)
extrema(dat)

dat = clamp(dat, 0.001, 0.999)
extrema(dat)

mean( diff(dat,1) .== 0. ) # pas de mouv dans 80% des cas


############  model

Nh = 30  # number of hidden units
Ne = 5   # number of mixture elements

pars₀ = Pars(Nh, Nd, Ne)
mean( xloglik(dat[:,j], pars₀)[1] for j in 1:Ns )


pars = sgd(pars₀, xloglik, xdloglik!, dat,
           maxtime=120, chunksize=100,
           kscale=1e-1, cbinterval=25, k0=1e-3)
mean( xloglik(dat[:,j], pars)[1] for j in 1:Ns )

pars = sgd(pars, xloglik, xdloglik!, dat,
           maxtime=300, chunksize=100,
           kscale=2e-2, cbinterval=50, k0=1e-3)
sum( xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2) )  # -107k

pars = sgd(pars, xloglik, xdloglik!, dat,
           maxsteps=1e5, maxtime=1000, chunksize=100,
           kscale=5e-3, cbinterval=50, k0=1e-3)
sum( xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2) )  # -107k



##############  drawing

sum( xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2) )
spars = deepcopy(pars)

## make a sample following the partial values given
xs = ones(10)*0.5
function xsample(xs::Vector{Float64}, pars::Pars)
  nx = length(xs)
  a  = copy(pars.c)
  ll  = 0.
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
    a  .+= pars.W[:,i] * xs[i]
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
      xs2[i] = clamp( xs2[i-1], 0.001, 0.999)
    else
      xs2[i] = ficdf3(cpars[ci,1], cpars[ci,2], rand())
    end

    xt[i] = loglik(cpars, xs2[i], clamp(xs2[i-1],0.001,0.999))
    ll += xt[i]
    a  .+= pars.W[:,i] * xs2[i]
  end


  xs2, xt, ll
end

xs2, lls, ll = xsample([0.], spars)

#### proba ponctuelle
nbcurves = 5
xs = linspace(0., 1., Nd)
px = xs * ones(nbcurves)'
py = similar(px)
cn=repeat([1:nbcurves;], inner=[Nd])

for i in 1:Ne-1 # i = 1
  py[:,i] = fk3.(xs, cpars[i,1], cpars[i,2])
end
py[:,end] = py[:,1:Ne-1] * wn[1:4]

data_values(x=vec(px), y=vec(py), cn=cn) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_nominal(:cn)


fk3

#### à partir de t₀ = 0
nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples)'
py = similar(px)

for i in 1:nbexamples
  py[:,i] = xsample([0.], spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_nominal(:cn)

# moyenne

px = repeat(linspace(0., 1., Nd), outer=2)
pm = zeros(Nd)

pmref = vec(mapslices(mean, dat, 2)) # moyenne training set

nbexamples = Ns
for i in 1:nbexamples
  pm .+= xsample([dat[2,i]], spars)[1]
end
pm ./= nbexamples

data_values(x=px, y=[pm; pmref], cn=repeat(["simul", "real"], inner=Nd)) +
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
