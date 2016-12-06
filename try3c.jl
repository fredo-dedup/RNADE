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
