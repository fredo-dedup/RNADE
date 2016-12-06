#########   fit de mixture #####################################
using VegaLite
using Distributions


data_values(x=collect(0:0.01:1.), y= fk3.(collect(0:0.01:1.), m, n)) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)

Ns = 500
dg = MixtureModel( [Beta(1,2), Beta(4,3), Beta(0.5,0.2)], [0.3,0.1,0.6] )
ys = rand(dg, Ns)

v = data_values(y=ys) + mark_tick() + encoding_x_quant(:y)

xs = collect(0.01:0.01:0.99)
data_values(x=xs, y=pdf(dg, xs)) +
    mark_line() +
    encoding_x_quant(:x)
    encoding_y_quant(:y)


Ne = 5  # nb of mixture elements
pars = rand(Normal(), Ne,3)
sum(loglik(pars, y) for y in ys)


dpars = zeros(pars)
    for y in ys
        dpars .+= dloglik(pars, y)
    end
    pars -= dpars * 0.01
    sum(loglik(pars, y) for y in ys)


xs = collect(0.01:0.01:0.99)
data_values(x=xs, y=map(x->exp(-loglik(pars,x)), xs)) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)

data_values(x=xs, y=pdf(dg, xs)) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)


##########################################################

Ns = 500
dg = MixtureModel( [Beta(10,20), Beta(50,30)], [0.3,0.7] )
ys = rand(dg, Ns)


Ne = 5  # nb of mixture elements
pars = rand(Normal(), Ne,3)
pars = [ linspace(-10,0,Ne) 10*ones(Ne) ones(Ne)  ]
pars = [ -6. 10. 1. ;
         -4. 10. 1. ]

sum(loglik(pars, y) for y in ys)

dpars = zeros(pars)
    for y in ys
        dpars .+= dloglik(pars, y)
    end
    pars -= dpars * 0.0001
    sum(loglik(pars, y) for y in ys)

println(dpars)
pars1 = sgd(pars, loglik, dloglik, ys, maxtime=15,
            kscale=1e-3, maxsteps=10000)

pars1 = sgd(pars1, loglik, dloglik, ys, maxtime=15,
            kscale=1e-3, maxsteps=10000)


function sgd(pars0, f, df, dat;
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(dat)
    datstate = start(datiter)

    pars = copy(pars0)
    dpars = zeros(pars)

    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        fill!(dpars, 0.)
        for i in 1:chunksize
            y, datstate = next(datiter, datstate)
            dpars .+= df(pars, y)
        end
        pars -= dpars * α * kscale
        α = 1. / (1 + t*k0)

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll = sum(f(pars, y) for y in dat)
            println("$t : α = $(round(α,3)) : $(round(ll,0))")
        end
    end

    pars
end


xs = collect(0.01:0.01:0.99)
data_values(x=xs, y=map(x->exp(-loglik(pars,x)), xs)) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)

data_values(x=xs, y=map(x->exp(-loglik(pars1,x)), xs)) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)



data_values(x=xs, y=pdf(dg, xs)) +
    mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)
