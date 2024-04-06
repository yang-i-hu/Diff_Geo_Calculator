from sympy import Matrix, sqrt
from sympy.abc import u, v, c
from sympy.vector import CoordSys3D
from sympy import symbols, cos, sin, diff, simplify, Rational


class ParametricSurface:
    def __init__(self, x):
        self.x = x
        self.x_u = self.x.diff(u)
        self.x_v = self.x.diff(v)
        self.normal = self.calculate_normal()
        self.E = self.calculate_E()
        self.F = self.calculate_F()
        self.G = self.calculate_G()
        self.x_uu = self.x_u.diff(u)
        self.x_uv = self.x_u.diff(v)
        self.x_vv = self.x_v.diff(v)
        self.l = self.calculate_l()
        self.m = self.calculate_m()
        self.n = self.calculate_n()
        self.K = self.calculate_K()
        self.H = self.calculate_H()

    def calculate_normal(self):
        normal = self.x_u.cross(self.x_v)
        return normal / sqrt(normal.dot(normal))

    def calculate_E(self):
        return simplify(self.x_u.dot(self.x_u))

    def calculate_F(self):
        return simplify(self.x_u.dot(self.x_v))

    def calculate_G(self):
        return simplify(self.x_v.dot(self.x_v))

    def calculate_l(self):
        return simplify(self.x_uu.dot(self.normal))

    def calculate_m(self):
        return simplify(self.x_uv.dot(self.normal))

    def calculate_n(self):
        return simplify(self.x_vv.dot(self.normal))

    def calculate_K(self):
        return simplify((self.l * self.n - self.m ** 2) / (self.E * self.G - self.F**2))

    def calculate_H(self):
        return simplify((self.E * self.n + self.G * self.l - 2 * self.F * self.m) / (2 * (self.E * self.G - self.F ** 2)))


    def christoffel_symbols(self):
        symbols = {}

        # Define the inverse of the First Fundamental Form matrix
        inv_first_fund_form = Matrix([[self.E, self.F], [self.F, self.G]]).inv()

        # Define the derivatives of the coefficients of the First Fundamental Form
        E_u, F_u, G_u = self.E.diff(u), self.F.diff(u), self.G.diff(u)
        E_v, F_v, G_v = self.E.diff(v), self.F.diff(v), self.G.diff(v)

        # Compute the components of the Christoffel symbols
        symbols['Γ^u_{uu}'], symbols['Γ^v_{uu}'] = inv_first_fund_form * Matrix(
            [Rational(1, 2) * E_u, F_u - Rational(1, 2) * E_v])
        symbols['Γ^u_{uv}'], symbols['Γ^v_{uv}'] = inv_first_fund_form * Matrix(
            [Rational(1, 2) * E_v, Rational(1, 2) * G_u])
        symbols['Γ^u_{vv}'], symbols['Γ^v_{vv}'] = inv_first_fund_form * Matrix(
            [F_u - Rational(1, 2) * G_u, Rational(1, 2) * G_v])

        return symbols

    def report(self):
        report_str = ""

        report_str += "Tangent vectors: \n"
        report_str += "x_u = " + str(self.x_u) + "\n"
        report_str += "x_v = " + str(self.x_v) + "\n\n"

        report_str += "Normal vector: \n"
        report_str += str(self.normal) + "\n\n"

        report_str += "Second derivatives: \n"
        report_str += "x_uu = " + str(self.x_uu) + "\n"
        report_str += "x_uv = " + str(self.x_uv) + "\n"
        report_str += "x_vv = " + str(self.x_vv) + "\n\n"

        report_str += "First Fundamental Form: E = " + str(self.E) + ", F = " + str(self.F) + ", G = " + str(self.G) + "\n"
        report_str += "Second Fundamental Form: l = " + str(self.l) + ", m = " + str(self.m) + ", n = " + str(self.n) + "\n\n"

        report_str += "Gaussian curvature: K = " + str(self.K) + "\n"
        report_str += "Mean curvature: H = " + str(self.H) + "\n"

        report_str += "\nChristoffel Symbols:\n"
        for key, value in self.christoffel_symbols().items():
            report_str += key + " = " + str(value) + "\n"

        return report_str

u, v = symbols('u v')
c, a, b = symbols('c a b')
r = symbols('r')
N = CoordSys3D('N')

x_shpere = c*sin(u) * cos(v) * N.i + c* sin(u) * sin(v) * N.j + c* cos(u) * N.k

x_stereo_sphere = 2*c*(u / (1 + u**2 + v**2)) * N.i + 2*c*(v / (1 + u**2 + v**2)) * N.j + c*((u**2 + v**2 - 1) / (1 + u**2 + v**2)) * N.k

x_cone = u * cos(v) * N.i + u * sin(v) * N.j + u * N.k

x_plane = u * cos(v) * N.i + u * sin(v) * N.j + v * N.k

x_helicoid = u * cos(v) * N.i + u * sin(v) * N.j + c * u * N.k

x_cylinder = r * cos(u) * N.i + r * sin(u) * N.j + v * N.k

x_saddle = u * N.i + v * N.j + u*v * N.k

x_torus =  (a+b*cos(u)) * cos(v) * N.i + (a+b*cos(u))  * sin(v) * N.j + b*sin(u) * N.k

x_Enneper = (u - u**3/3 + u*v**2) * N.i + (v - v**3/3 + v*u**2) * N.j + (u**2 - v**2) * N.k

print(ParametricSurface(x_cylinder).report())