using ReverseDiffSource

ek3 = quote
    em = exp(m)
    a = sqrt(em * exp(n))
    b = a / em
    a*b*x^(a-1.) * (1.-x^a)^(b-1.)
end

@eval fk3(x::Float64,m::Float64,n::Float64) = $ek3
edk3 = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x, allorders=false)
@eval fdk3(x::Float64,m::Float64,n::Float64) = $edk3
edk3a = rdiff(ek3, x = Float64, m=Float64, n=Float64, ignore=:x)
@eval fdk3a(x::Float64,m::Float64,n::Float64) = $edk3a

m, n = -1., 2.0
fk3(0.5, m, n)
fdk3(0.5, m, n)

δ = 1e-8
(fk3(0.5, m+δ, n)-fk3(0.5, m, n))/δ
(fk3(0.5, m, n+δ)-fk3(0.5, m, n))/δ


xs = 0:0.01:1.
data_values(x=xs, y=fk3.(xs, -3., 10.)) +
      mark_line() + encoding_x_quant(:x) + encoding_y_quant(:y)

function fcdf3(m::Float64,n::Float64,x::Float64)
  em = exp(m)
  a = sqrt(em * exp(n))
  b = a / em
  1. - (1 - x^a)^b
end


fcdf3(-5.,10.,0.5)

fcdf3.(-2.,2.,[0:0.1:1;])

function ficdf3(m::Float64,n::Float64,p::Float64)
  em = exp(m)
  a = sqrt(em * exp(n))
  b = a / em
  (1. - (1. - p)^(1/b))^(1/a)
end

fcdf3(0.,2.,0.5)
ficdf3(0.,2.,0.361)
ficdf3(0.,2.,0.5)
