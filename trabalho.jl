using Random, Distributions, ProgressBars

function dU(T, U0, T1, T2)
    2 * U0 * (T - T1) * (T - T2) * (2 * T - T1 - T2) / (T1 - T2)^4
end

τ = 1e5
ω = 2pi / τ
T1 = 280.0
T2 = 290.0
U0 = 213.0
C = 2pi * (T1 - T2)^2 / (sqrt(16*32) * U0)
gg = sqrt(2 * U0 / 2)
ϵ = 20

print("C = ", C)
print('\n')
print("g = ", gg)
print('\n')

dt = 0.01
t_f = 300.0 * 1e3
n = Integer(t_f ÷ dt) - 1
t = dt * Array(range(0, n - 1))

T = Array(zeros(n))
T[1] = T1

dW = Array(rand(Normal(0, sqrt(dt)), n))

for i in ProgressBar(1:n - 1)
    local  F_ii = - dU(T[i], U0, T1, T2) - ϵ * cos(ω * t[i])
    global T[i + 1] = T[i] + dt * F_ii + gg * dW[i]
end
