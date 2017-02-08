#### using bernstein polynomials to approximate the Distribution

using VegaLite
using Distributions
using ReverseDiffSource

###############  plotting ######################

function plotm(xp::Vector{Float64}, dys::Dict{Symbol,Vector{Float64}})
  # dys = Dict(:prob => map(fp, xp))
  rxp = repeat(xp, outer=length(dys))
  rys = vcat(values(dys)...)
  rls = repeat(collect(keys(dys)), inner=length(xp))
  data_values(x=rxp, y=rys, name=rls) +
    encoding_x_quant(:x) +
    encoding_y_quant(:y) +
    encoding_color_nominal(:name) +
    mark_line()
end

function plotm(xp::Vector{Float64}, dyf::Dict)
  dys = Dict( lab => map(f,xp) for (lab,f) in dyf)
  plotm(xp, dys)
end

plotm(xp, Dict(:prob0 => x -> pdf(Beta(2,3),x)))

####################################################

N = 10
w = rand(Normal(), N)
cw = [binomial(N-1,i) for i in 0:N-1]

ep = quote
  nw = log(1. .+ exp(w))
  fnw = N / sum(nw)
  vs = zeros(N)
  for i in 1:N
    vs[i] = x^(i-1) * (1. - x)^(N-i)
  end
  cs = nw .* cw
  dot(cs, vs) * fnw
end

@eval fp(x) = $ep  # prob function
fp(0.5)
fp(0.)

quadgk(fp, 0., 1.)

xp = collect(linspace(0,1,100))
plotm(xp, Dict(:prob => map(fp, xp)))

@eval fl(x) = log($ep)  # ll function
fl(0.5)

edl = rdiff(:(log($ep)), allorders=false, x=Float64, w=Vector{Float64}, ignore=[:x])
@eval fdl(x) = $edl # diffs of loglik function

# x₀ = 0.3
# dw = fdl(x₀)
# v0 = fl(x₀)
#
# δ = 1e-8
# w[10] += δ
# (fl(x₀)-v0) / δ
# dw[10]

function fulldp(xs::Vector{Float64})
    dw = zeros(w)
    for x in xs
      dw += fdl(x)
    end
    dw
end

fulldp(xs)


# xs = rand(Beta(2,3),500)
dxs = MixtureModel([Beta(4,6), Beta(10,2)])
xs = rand(dxs,500)
fxs(x) = pdf(dxs,x)

N = 21
w = rand(Normal(), N)
cw = [binomial(N-1,i) for i in 0:N-1]
mean( fl(x) for x in xs )

μ = 1e-1
  dw = fulldp(xs)
  ρ = 1. # 1 / norm(dw)
  w += μ * min(1., ρ) * dw
  mean( fl(x) for x in xs )

norm(dw)


xp = collect(linspace(0, 1, 100))
plotm(xp, Dict(:target => fxs, :result => fp))


#### convergence par sgd  #######################

dxs = MixtureModel([Beta(4,6), Beta(10,2)])
xs = rand(dxs,500)
fxs(x) = pdf(dxs,x)
xst = rand(dxs,500)


N
w
@eval fl2(x,w) = log($ep)
@eval fdl2(x,w) = $edl

fl2(0.2,w)
fdl2(0.2,w)

function sgd(pars₀;
             dat  = xs,
             datt = xst,
             maxtime=10, maxsteps=1000, chunksize=100,
             kscale=1e-4, cbinterval=100, k0=1e-3)

    # pars₀ = w ; dat = xs

    α    = 1.      # acceleration parameter
    starttime = time()
    datiter = cycle(1:length(dat))
    datstate = start(datiter)

    pars   = deepcopy(pars₀)
    dpars  = deepcopy(pars)

    # t = 1
    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        fill!(dpars, 0.)
        # i = 1
        for i in 1:chunksize
            yi, datstate = next(datiter, datstate)
            dpars += fdl2(dat[yi], pars)
        end
        # dmax = maximum(dpars)
        # (100./dmax < α*kscale/chunksize) && print("+")
        # scal!(dpars, - min(100./dmax, α*kscale/chunksize))
        # clamp!(scal!(dpars, -α*kscale/chunksize), -1., 1.)
        pars += α*kscale/chunksize * dpars
        α = 1. / (1 + t*k0)

        if any(isnan(pars))
          break
        end

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            ll  = mean( fl2(x,pars) for x in dat ) # score(pars,  dat, f)
            llt = mean( fl2(x,pars) for x in datt ) # score(pars, datt, f)
            println("$t : α = $(round(α,3)), train : $(round(ll,6)), test : $(round(llt,6))")
        end
    end

    pars
end


N = 5
w = rand(Normal(), N)
cw = [binomial(N-1,i) for i in 0:N-1]
mean( fl2(x,w) for x in xs )

w1 = sgd(w, kscale=5., chunksize=95, maxsteps=1e4)

plotm(xp, Dict(:target => fxs, :result => x -> exp(fl2(x,w1))) )
