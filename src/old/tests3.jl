using VegaLite

function sgd(pars₀, f, df!, dat;
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    f = xloglik
    df! = xdloglik!

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
            ll = sum( f(dat[:,j], pars)[1] for j in 1:size(dat,2) )
            println("$t : α = $(round(α,3)) : $(round(ll,0))")
        end
    end

    pars
end


###########  samples

Ns = 1000
Nd = 50  # variable size to estimate

x1 = [ sin(π * x) for x in linspace(0., 1., Nd) ]
x2 = [ x for x in linspace(0., 1., Nd) ]
typs = rand(Bool,Ns)
dat = hcat( [ t ? x1 : x2 for t in typs ]... )
dat = clamp( dat + rand(Normal(0.,0.02), size(dat)), 0.001, 0.999 )
extrema(dat)

############  model

Nh = 10  # number of hidden units
Ne = 5   # number of mixture elements

pars₀ = Pars(Nh, Nd, Ne)
sum( xloglik(dat[:,j], pars₀)[1] for j in 1:size(dat,2) )


pars = sgd(pars₀, xloglik, xdloglik!, dat, maxtime=60, maxsteps=1000, chunksize=100,
           kscale=1e-1, cbinterval=10, k0=1e-2)

pars = sgd(pars, xloglik, xdloglik!, dat, maxtime=60, maxsteps=1000, chunksize=100,
           kscale=1e-3, cbinterval=10, k0=1e-2)

pars = sgd(pars, xloglik, xdloglik!, dat, maxtime=300, maxsteps=1000, chunksize=100,
           kscale=2e-2, cbinterval=10, k0=1e-2)

pars = sgd(pars, xloglik, xdloglik!, dat, maxtime=60, maxsteps=1000, chunksize=100,
          kscale=1e-3, cbinterval=10, k0=1e-2)

pars = sgd(pars, xloglik, xdloglik!, dat, maxtime=60, maxsteps=1000, chunksize=100,
           kscale=3e-2, cbinterval=10, k0=1e-2)

pars = sgd(pars, xloglik, xdloglik!, dat, maxtime=100, maxsteps=1000, chunksize=100,
          kscale=1e-6, cbinterval=1, k0=1e-1)



##############  drawing

sum( xloglik(dat[:,j], pars)[1] for j in 1:size(dat,2) )
spars = deepcopy(pars)

## make a sample following the partial values given
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
      m = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      n = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      w = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
      cpars[j,:] = [m,n,w]
    end

    xt[i] = loglik(cpars, xs[i])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs[i]
  end

  xs2 = zeros(Nd)
  xs2[1:nx] = xs
  # sampled part
  for i in nx+1:Nd # i = 1
    h[:,i] .= sigm.(a)

    for j in 1:Ne
      m = pars.bm[j][i] + dot(pars.Vm[j][:,i], h[:,i])
      n = pars.bn[j][i] + dot(pars.Vn[j][:,i], h[:,i])
      w = pars.bw[j][i] + dot(pars.Vw[j][:,i], h[:,i])
      cpars[j,:] = [m,n,w]
    end

    # pick component
    wn = exp.(cpars[:,3])
    wn ./= sum(wn)
    ci = rand(Categorical(wn))

    # pick x value
    xs2[i] = ficdf3(cpars[ci,1], cpars[ci,2], rand())

    xt[i] = loglik(cpars, xs2[i])
    ll += xt[i]
    a  .+= pars.W[:,i] * xs2[i]
  end


  xs2, xt, ll
end

xs2, lls, ll = xsample([0.], spars)


#### à partir de t₀ = 0
nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples+2)'
py = similar(px)

py[:,1] = x1
py[:,2] = x2
for i in 3:nbexamples+2
  py[:,i] = xsample([0.], spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples+2;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)

#### à partir de t₀ = 0.4, sur profil de x2
xs0 = x2[1:20]

nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples+2)'
py = similar(px)

py[:,1] = x1
py[:,2] = x2
for i in 3:nbexamples+2
  py[:,i] = xsample(xs0, spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples+2;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)


#### à partir de t₀ = 0.4, sur profil plat
xs0 = repeat([0.4;], inner=20)

nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples+2)'
py = similar(px)

py[:,1] = x1
py[:,2] = x2
for i in 3:nbexamples+2
  py[:,i] = xsample(xs0, spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples+2;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)

#### à partir de t₀ = 0.4, sur profil x1
xs0 = x1[1:20]

nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples+2)'
py = similar(px)

py[:,1] = x1
py[:,2] = x2
for i in 3:nbexamples+2
  py[:,i] = xsample(xs0, spars)[1]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples+2;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)


###########



nbexamples = 10
px = linspace(0., 1., Nd) * ones(nbexamples+2)'
py = similar(px)

py[:,1] = x1
py[:,2] = x2
for i in 3:nbexamples
  py[:,i] = dat[:, rand(1:Ns)]
end

data_values(x=vec(px), y=vec(py), cn=repeat([1:nbexamples+2;], inner=[Nd])) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y) +
      encoding_color_ord(:cn)
