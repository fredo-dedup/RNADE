################################################################
#
#  setup 15 (optim allocs et calculs du 14)
#  kuma a,b = 1 + log(1+exp(-x)), + ponctuel 0 , x et 1.
#  activation sigmoide
#  avec pénalisation
#
################################################################

using Distributions
# using BenchmarkTools
using VegaLite

@time for i in 1:1e6 ; 3.5 ^ 1.123 ; end  # 0.106s
@time for i in 1:1e6 ; exp(log(3.5) * 1.123) ; end # 0.05 plus rapide !!!

############## internal model specs  #############################

# type inference does not work ^ => creation of a power function

module Model
  using ReverseDiffSource

  export loglik, dloglik!, ficdf3

  #### elementary model
  ek3 = quote
      a = 1. + log(1+exp(m))
      b = 1. + log(1+exp(n))
      a*b*exp(log(x)*(a-1.)) * exp(log(1. - exp(log(x)*a))*(b-1.))
  end

  # TODO : optim du code généré

  @eval fk3(x::Float64,m::Float64,n::Float64) = $ek3
  edk3 = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x, allorders=false)
  @eval fdk3(x::Float64,m::Float64,n::Float64) = $edk3
  edk3a = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x)
  @eval fdk3a(x::Float64,m::Float64,n::Float64) = $edk3a

  δ = 1e-8
  # pars1 = copy(pars) ; pars1[1,1] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # pars1 = copy(pars) ; pars1[1,2] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # pars1 = copy(pars) ; pars1[1,3] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # dloglik(pars, 0.9)[6,:]

  function fcdf3(m::Float64,n::Float64,x::Float64)
    a = 1. + log(1+exp(m))
    b = 1. + log(1+exp(n))
    1. - (1 - x^a)^b
  end

  # fcdf3(-5.,10.,0.5)
  # fcdf3.(-2.,2.,[0:0.1:1;])

  function ficdf3(m::Float64,n::Float64,p::Float64)
    a = 1. + log(1+exp(m))
    b = 1. + log(1+exp(n))
    (1. - (1. - p)^(1/b))^(1/a)
  end

  # fcdf3(0.,2.,0.5)
  # ficdf3(0.,2.,0.361)
  # ficdf3(0.,2.,0.5)
  ficdf3(-1.,-1.,0.5)

  ####### loglik definitions  #######################################

  const pwidth = 1e-2
  function loglik(cpars::Matrix{Float64}, x::Float64, x₀::Float64)
      # x, x₀ = 0.5, 0.3
      ne = size(cpars,1)
      ws = exp.(cpars[:,3])
      ws ./= sum(ws)

      pm = 0.
      for i in 1:ne-3
          pm += ws[i] * fk3(x, cpars[i,1], cpars[i,2])
      end

      pw0, pw1, hpw, ipw = pwidth, 1. - pwidth, pwidth / 2., 1. / pwidth
      x00 = clamp(x₀ - hpw, 0. , pw1)
      x01 = clamp(x₀ + hpw, pw0, 1. )
      x00 < x < x01 && (pm += ws[ne-1]*ipw)
      x>pw1 && (pm += ws[ne]*ipw)
      x<pw0 && (pm += ws[ne-2]*ipw)

      -log(pm)
  end
  # loglik(pars, 0.5, 0.5)
  # exp(-loglik(pars, 0.1))
  #
  # @benchmark loglik(pars, 0.5) # 28us
  #
  # cpars = rand(Normal(), Ne,3)
  # loglik(cpars,0.5,0.2)
  #
  # i = 5
  # quadgk( x -> fk3(x, cpars[i,1], cpars[i,2]), 0., 1.)
  #
  # px = linspace(0.0001, 0.9999, 10000.)
  # py = map( x -> exp(-loglik(cpars, x, 1.0)), px)
  # mean(py)
  #
  # data_values(x=px, y=py) + mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)


  # delta = 1e-4
  # pars1 = copy(pars) ; pars1[1,1] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # pars1 = copy(pars) ; pars1[1,2] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # pars1 = copy(pars) ; pars1[1,3] += delta
  # (loglik(pars1,0.9)-loglik(pars,0.9))/delta
  #
  # dloglik(pars, 0.9)[6,:]

  x, x₀ = 0.5, 0.3
  function dloglik!(cpars::Matrix{Float64}, dcpars::Matrix{Float64},
                    x::Float64, x₀::Float64)
      # p = 0.5
      # x, x₀ = 0.5, 0.5
      ne = size(cpars,1)
      ws0 = exp.(cpars[:,3])
      sws0 = sum(ws0)
      ws = ws0 ./ sws0
      ps = Array(Float64, ne)
      for i in 1:ne-3
        ps[i] = fk3(x, cpars[i,1], cpars[i,2])
      end

      pw0, pw1, hpw, ipw = pwidth, 1. - pwidth, pwidth / 2., 1. / pwidth
      x00 = clamp(x₀ - hpw, 0. , pw1)
      x01 = clamp(x₀ + hpw, pw0, 1. )
      ps[ne-1] = (x00 < x < x01) * ipw
      ps[ne]   = (x > pw1) * ipw
      ps[ne-2] = (x < pw0) * ipw

      pm = dot(ws, ps)

      _tmp5 = - ps ./ pm
      dcpars[:,3] = ws0 ./ sws0 .* ( _tmp5 + ( sum(-ws0 .* _tmp5) / sws0 ) )

      dv0 = -1/pm
      for i in 1:ne-3
          dcpars[i,1], dcpars[i,2] = fdk3(x, cpars[i,1], cpars[i,2])
          dcpars[i,1] *= ws[i] * dv0
          dcpars[i,2] *= ws[i] * dv0
      end
      dcpars[ne-1,1], dcpars[ne-1,2] = 0., 0.
      dcpars[ne  ,1], dcpars[ne  ,2] = 0., 0.
      dcpars[ne-2,1], dcpars[ne-2,2] = 0., 0.

      # any(isnan(dcpars)) ? zeros(cpars) : dcpars
  end

  #####  testing
  if false
    function dtest(cpars, x, x₀, indexes)
      # field, indexes = :Vm, [2, [1,1]]
      δ = 1e-8

      p0 = getindex(cpars, indexes...)
      cpars2 = deepcopy(cpars)
      setindex!(cpars2, p0+δ, indexes...)

      ed = (loglik(cpars2, x, x₀) - loglik(cpars, x, x₀)) / δ

      dcpars = zeros(cpars)
      dloglik!(cpars, dcpars, x, x₀)
      ed0 = getindex(dcpars, indexes...)
      ( ed0, ed )
    end

    cpars = rand(5,3)

    dtest(cpars, 0.5, 0.4, [1,1])
    dtest(cpars, 0.5, 0.4, [2,2])

    dtest(cpars, 0.5, 0.4, [5,1])
    dtest(cpars, 0.5, 0.4, [5,2])
    dtest(cpars, 0.5, 0.4, [3,1])
    dtest(cpars, 0.5, 0.4, [4,2])

    dtest(cpars, 0.5, 0.4, [1,3])
    dtest(cpars, 0.5, 0.4, [5,3])
    dtest(cpars, 0.5, 0.4, [5,3])

    dtest(cpars, 0.5, 0.5, [1,1])
    dtest(cpars, 0.5, 0.5, [2,2])

    dtest(cpars, 0.5, 0.5, [1,3])
    dtest(cpars, 0.5, 0.5, [3,3])
    dtest(cpars, 0.5, 0.5, [4,3])
    dtest(cpars, 0.5, 0.5, [5,3])

    dtest(cpars, 0.0001, 0.5, [1,1])
    dtest(cpars, 0.0001, 0.5, [2,2])

    dtest(cpars, 0.0001, 0.5, [1,3])
    dtest(cpars, 0.0001, 0.5, [3,3])
    dtest(cpars, 0.0001, 0.5, [4,3])
    dtest(cpars, 0.0001, 0.5, [5,3])

    dtest(cpars, 0.0001, 0.000, [1,1])
    dtest(cpars, 0.0001, 0.000, [2,2])

    dtest(cpars, 0.0001, 0.000, [1,3])
    dtest(cpars, 0.0001, 0.000, [3,3])
    dtest(cpars, 0.0001, 0.000, [4,3])
    dtest(cpars, 0.0001, 0.000, [5,3])

    dtest(cpars, 0.9995, 0.000, [1,1])
    dtest(cpars, 0.9995, 0.000, [2,2])

    dtest(cpars, 0.9995, 0.000, [1,3])
    dtest(cpars, 0.9995, 0.000, [3,3])
    dtest(cpars, 0.9995, 0.000, [4,3])
    dtest(cpars, 0.9995, 0.000, [5,3])

    dtest(cpars, 0.9995, 1., [1,1])
    dtest(cpars, 0.9995, 1., [2,2])

    dtest(cpars, 0.9995, 1., [1,3])
    dtest(cpars, 0.9995, 1., [3,3])
    dtest(cpars, 0.9995, 1., [4,3])
    dtest(cpars, 0.9995, 1., [5,3])
  end

end

############## RNADE param type definition ####################

module RNADE
  using Distributions
  # using Model
  using ..Model

  #### types
  begin
    type Pars{Nh,Nd,Ne}
      Vs::Vector{Matrix{Float64}}  # for m mixture component
      bs::Vector{Vector{Float64}}  # for m mixture component
      W::Matrix{Float64}
      c::Vector{Float64}
    end

    function Pars(Nh, Nd, Ne)
        Pars{Nh,Nd,Ne}([rand(Normal(), 3Ne, Nh) for i in 1:Nd],
                       [rand(Normal(), 3Ne) for i in 1:Nd],
                       rand(Normal(), Nh, Nd),
                       rand(Normal(), Nh)
        )
    end

    function add!{Nh,Nd,Ne}(a::Pars{Nh,Nd,Ne}, b::Pars{Nh,Nd,Ne})
      for i in 1:Nd
        a.Vs[i] .+= b.Vs[i]
        a.bs[i] .+= b.bs[i]
      end
      a.W .+= b.W
      a.c .+= b.c
      a
    end

    function scal!{Nh,Nd,Ne}(a::Pars{Nh,Nd,Ne}, f::Float64)
      for i in 1:Nd
        a.Vs[i] .*= f
        a.bs[i] .*= f
      end
      a.W .*= f
      a.c .*= f
      a
    end

    import Base.clamp!
    function clamp!{Nh,Nd,Ne}(a::Pars{Nh,Nd,Ne}, low::Float64, up::Float64)
      for i in 1:Nd
        clamp!(a.Vs[i], low, up)
        clamp!(a.bs[i], low, up)
      end
      clamp!(a.W, low, up)
      clamp!(a.c, low, up)
      a
    end

    import Base.maximum
    function maximum{Nh,Nd,Ne}(a::Pars{Nh,Nd,Ne})
      mx = -Inf
      for i in 1:Nd
        mx = max(mx, maxabs(a.Vs[i]))
        mx = max(mx, maxabs(a.bs[i]))
      end
      mx = max(mx, maxabs(a.W))
      mx = max(mx, maxabs(a.c))
      mx
    end

    function zeros!{Nh,Nd,Ne}(a::Pars{Nh,Nd,Ne})
      for i in 1:Nd
        fill!(a.Vs[i], 0.)
        fill!(a.bs[i], 0.)
      end
      fill!(a.W, 0.)
      fill!(a.c, 0.)
      a
    end
  end


  ############  RNADE defs #######################################

  sigm(x::Float64) = 1 ./ (1+exp(-x))
  const llp_fac = 0.
  # xs = test_set[:,225]

  function xloglik{Nh,Nd,Ne}(xs::Vector{Float64}, pars::Pars{Nh,Nd,Ne})
    nx = length(xs)
    a  = copy(pars.c)
    ll  = 0.
    xt = Array(Float64, nx)
    h  = Array(Float64, Nh, nx)
    cpars = Array(Float64, Ne, 3)
    for i in 1:nx # i = 1
      h[:,i] .= sigm.(a)
      # h[:,i] .= max.(0., a)

      cpars = reshape(pars.bs[i] .+ pars.Vs[i] * h[:,i], Ne, 3)

      xt[i] = loglik(cpars, xs[i], (i==1) ? xs[1] : xs[i-1])
      ll += xt[i]
      a  .+= pars.W[:,i] * xs[i]
    end

    # penalisation
    # llp = 0.
    # for i in 1:Nd
    #   llp += dot(pars.Vs[i],pars.Vs[i])
    #   llp += dot(pars.bs[i],pars.bs[i])
    # end
    # llp += dot(pars.W,pars.W)
    # llp += dot(pars.c,pars.c)
    #
    # ll += llp_fac * llp

    ll, xt, h
  end

  pars = Pars(10, 50, 7)
  xloglik(rand(30), pars)

  dpars = deepcopy(pars)

  let

    global xdloglik!, init, xsample

    # Nh, Nd, Ne = 10, 50, 7
    local a::Vector{Float64}, h::Vector{Vector{Float64}}
    local vcpars::Vector{Vector{Float64}}
    local δh::Vector{Float64}, δa::Vector{Float64}
    local dcpars::Matrix{Float64}

    function init{Nh,Nd,Ne}(pars::Pars{Nh,Nd,Ne})
      a  = Array(Float64, Nh)
      h  = [ Array(Float64, Nh) for i in 1:Nd ]
      vcpars = [ Array(Float64, 3Ne) for i in 1:Nd]
      δh = Array(Float64, Nh)
      δa = similar(a)
      dcpars = Array(Float64, Ne, 3)
    end

    function xdloglik!{Nh,Nd,Ne}(xs::Vector{Float64},
                                 pars::Pars{Nh,Nd,Ne},
                                 dpars::Pars{Nh,Nd,Ne}) # x, pars = x₀, pars₀
      # Nh, Nd, Ne = 10, 50, 7
      nx = length(xs)
      copy!(a, pars.c)
      for i in 1:nx # i = 1
        for j in 1:Nh ; h[i][j] = sigm(a[j]) ; end
        # h[:,i] .= max.(0., a)

        A_mul_B!(vcpars[i], pars.Vs[i], h[i])
        vcpars[i] .+= pars.bs[i]

        a  .+= pars.W[:,i] * xs[i]
      end

      zeros!(dpars)

      # δh = Array(Float64, Nh)
      fill!(δa, 0.)

      dcpars = Array(Float64, Ne, 3)
      for i in nx:-1:1 # i = nx-1
        cpars = reshape(vcpars[i], Ne, 3)
        dloglik!(cpars, dcpars, xs[i], (i==1) ? xs[1] : xs[i-1])

        copy!(dpars.bs[i], vec(dcpars))
        A_mul_Bt!(dpars.Vs[i], vec(dcpars), h[i])
        # dpars.Vs[i] = vec(dcpars) * h[i]'

        At_mul_B!(δh, pars.Vs[i], vec(dcpars))

        dpars.W[:,i] = δa * xs[i]
        δa += δh .* h[i] .* (1. - h[i])
        # δa += δh[:,i] .* (h[:,i] .> 0.)
      end
      copy!(dpars.c, δa)

      # penalisation
      # for i in 1:Nd
      #   dpars.Vs[i] += 2 * llp_fac .* pars.Vs[i]
      #   dpars.bs[i] += 2 * llp_fac .* pars.bs[i]
      # end
      # dpars.W += 2 * llp_fac .* pars.W
      # dpars.c += 2 * llp_fac .* pars.c

      dpars
    end

    function xsample{Nh,Nd,Ne}(xs::Vector{Float64}, pars::Pars{Nh,Nd,Ne})
      nx = length(xs)
      ll = 0.
      xt = Array(Float64, Nd)

      copy!(a, pars.c)

      # known part
      for i in 1:nx # i = 1
        for j in 1:Nh ; h[i][j] = sigm(a[j]) ; end

        A_mul_B!(vcpars[i], pars.Vs[i], h[i])
        vcpars[i] .+= pars.bs[i]

        prevxt = (i==1) ? xs[1] : xs[i-1]
        xt[i] = loglik(reshape(vcpars[i], Ne, 3), xs[i], prevxt)
        ll += xt[i]

        a  .+= pars.W[:,i] * xs[i]
      end

      xs2 = zeros(Nd)
      xs2[1:nx] = xs
      # sampled part
      for i in nx+1:Nd # i = nx+1
        h[i] .= sigm.(a)

        A_mul_B!(vcpars[i], pars.Vs[i], h[i])
        vcpars[i] .+= pars.bs[i]

        # pick component
        wn = exp.(vcpars[i][1+2Ne:end])
        wn ./= sum(wn)
        ci = rand(Categorical(wn))

        # pick x value
        if ci == Ne   # on 1.
            xs2[i] = 0.9999
        elseif ci == Ne-1 # centered on x(t-1)
            xs2[i] = xs2[i-1]
        elseif ci == Ne-2  # on 0.
            xs2[i] = 0.0001
        else
            xs2[i] = ficdf3(vcpars[i][ci], vcpars[i][ci+Ne], rand())
        end

        xt[i] = loglik(reshape(vcpars[i], Ne, 3), xs2[i], xs2[i-1])
        ll += xt[i]
        a  .+= pars.W[:,i] * xs2[i]
      end

      xs2, xt, ll
    end


  end

  pars = Pars(10, 50, 7)
  dpars = deepcopy(pars)
  init(pars)
  # whos(r"pars") # ~400kb

  xs = rand(20)

  xloglik(xs, pars)
  xdloglik!(xs, pars, dpars)
  dpars.c[1]
  dpars.c

  # profiling #################

  # Profile.clear()
  # Base.@profile collect(xdloglik!(rand(50), pars, dpars) for i in 50:500)
  # Profile.print()
  #
  # xs = rand(50)
  @time collect(RNADE.xdloglik!(rand(50), pars, dpars) for i in 50:500)
  # 3.83 s
  # 3.14 s
  # 3.00 s
  # 2.86 s
  # 1.81 s
  # 0.71 s
  # 0.29 s
  # 0.24 s
  # 0.22 0.27 s

  ###############  testing ########################

  if false
    pars = scal!(Pars(10, 50, 7), 0.1)
    dpars = deepcopy(pars); zeros!(dpars)
    xs = rand(20)

    # xs = Main.train_set[:,400]

    dpars = xdloglik!(xs, pars, dpars)
    v0 = xloglik(xs,pars)[1]

    function dtest(v0, pars, dpars, field, indexes...)
      # field, indexes = :Vm, [2, [1,1]]
      δ = 1e-8

      p0 = foldl((x,idx) -> getindex(x, idx...), getfield(pars, field), indexes)

      pars2 = deepcopy(pars)
      np = foldl((x,idx) -> getindex(x, idx...), getfield(pars2, field), indexes[1:end-1])
      setindex!(np, p0+δ, indexes[end]...)

      ed = (xloglik(xs,pars2)[1]-v0) / δ

      ed0 = foldl((x,idx) -> getindex(x, idx...), getfield(dpars, field), indexes)
      ( ed0, ed )
    end

    #### version de test sur un bloc d'exemples
    # function dtest(v0, pars, dpars, field, indexes...)
    #   # field, indexes = :Vm, [2, [1,1]]
    #   δ = 1e-8
    #   p0 = foldl((x,idx) -> getindex(x, idx...), getfield(pars, field), indexes)
    #
    #   pars2 = deepcopy(pars)
    #   np = foldl((x,idx) -> getindex(x, idx...), getfield(pars2, field), indexes[1:end-1])
    #   setindex!(np, p0+δ, indexes[end]...)
    #
    #   ## eval
    #   nv0 = 0.
    #   for yi in trg
    #       nv0 += xloglik(train_set[:,yi],pars2)[1]
    #   end
    #
    #   ed = (nv0-v0) / δ
    #
    #   ed0 = foldl((x,idx) -> getindex(x, idx...), getfield(dpars, field), indexes)
    #   ( ed0, ed )
    # end
    #
    # trg = 1:500
    # dparsi = deepcopy(pars)
    # dpars  = deepcopy(pars)
    # zeros!(pars)
    # zeros!(dpars)
    # for yi in trg
    #     add!(dpars, xdloglik!(train_set[:,yi], pars, dparsi) )
    # end
    #                                         dpars
    #
    # pars
    #
    # v0 = 0.
    # for yi in trg
    #     v0 += xloglik(train_set[:,yi],pars)[1]
    # end

    ####################

    nd = length(pars.Vs)
    res = Array(Float64, nd, 2)
    for i in 1:nd
      res[i,1], res[i,2] = dtest(v0, pars, dpars, :Vs, i, [1,1])
    end
    res

    for i in 1:nd
      res[i,1], res[i,2] = dtest(v0, pars, dpars, :bs, i, 1)
    end
    res

    for i in 1:nd
      res[i,1], res[i,2] = dtest(v0, pars, dpars, :bs, i, 10)
    end
    res

    dtest(v0, pars, dpars, :Vs, 2, [10,1])
    dtest(v0, pars, dpars, :Vs, 20, [1,10])
    dtest(v0, pars, dpars, :Vs, 50, [21,3])
    dtest(v0, pars, dpars, :Vs, 1, [10,10])

    dpars.bs[50]

    dtest(v0, pars, dpars, :bs, 2, 1)
    dtest(v0, pars, dpars, :bs, 2, 8)
    dtest(v0, pars, dpars, :bs, 2, 15)
    dtest(v0, pars, dpars, :bs, 2, 20)
    dtest(v0, pars, dpars, :bs, 2, 21)

    dtest(v0, pars, dpars, :bs, 50, 1)
    dtest(v0, pars, dpars, :bs, 50, 10)
    dtest(v0, pars, dpars, :bs, 50, 15)
    dtest(v0, pars, dpars, :bs, 50, 20)
    dtest(v0, pars, dpars, :bs, 50, 21)

    dtest(v0, pars, dpars, :W, [1,1])
    dtest(v0, pars, dpars, :W, [10,1])
    dtest(v0, pars, dpars, :W, [1,10])
    dtest(v0, pars, dpars, :W, [10,30])
    dtest(v0, pars, dpars, :W, [10,15])

    dtest(v0, pars, dpars, :c, 1)
    dtest(v0, pars, dpars, :c, 10)
    dtest(v0, pars, dpars, :c, 5)
    dtest(v0, pars, dpars, :c, 2)
    dtest(v0, pars, dpars, :c, 7)
    dtest(v0, pars, dpars, :c, 9)

  end

  ##########################################################################
  ## make a sample following the partial values given
  ##########################################################################

  xs = [0., 0., 0.;]



end

# wn = exp.(cpars[:,3])
# wn ./= sum(wn)
# ci = rand(Categorical(wn))
#
# # pick x value
# if ci == Ne   # on 1.
#     0.9999
# elseif ci == Ne-1 # centered on x(t-1)
#     xs2[i-1]
# elseif ci == Ne-2  # on 0.
#     0.0001
# else
#     ficdf3(cpars[ci,1], cpars[ci,2], rand())
# end
