from potto.lang.evaluate import evaluate
from potto.lang.derivative import deriv
from potto.lang.grammar import App, Const, Diffeomorphism, Function, GExpr, IfElse, TegVar, Var, Int
from potto.libs.pmath import Sqrt, Abs
from potto.libs.measure import BoundedLebesgue
from potto.lang.samples import VarVal
from potto.libs.diffeos import Affine
from scipy.optimize import minimize

x = TegVar("x")
y = TegVar("y")

c = Var("c")
dc = Var("dc")
color = Var("color")


def loss(c, color, image, shader):
    diffeo = Affine((1, 1, c), (x, y))
    mu = BoundedLebesgue(0, 1, x)
    nu = BoundedLebesgue(0, 1, x)
    mask = IfElse(diffeo, 1, 0)
    integrand = c * x + App(shader, (x, y, color)) * mask
    pixel_color = Int(Int(integrand, mu), nu)
    Function(shader, pixel_color)
    return (pixel_color - image) ** 2


const_shader = Function((x, y, color), color)
linear_shader = Function((x, y, color), x + y + color)


image = 0.7
translate_init = 0.5
shader = Var("shader")
dloss = deriv(
    loss(c, color, image, shader),
)

dconst_shader = deriv(const_shader, {c.name: dc.name})
dlin_shader = deriv(linear_shader)

cs = []
for dshader in [dconst_shader, dlin_shader]:
    var_val = VarVal({c.name: translate_init, color.name: 1})
    gradient = evaluate(dloss(dshader), var_val)
    cs.append(minimize(gradient, translate_init))


# # def half_space(x, y, a, b, c):
# #     tv0, tv1 = TegVar("tv0"), TegVar("tv1")
# #     a_, b_, c_ = Var("a_"), Var("b_"), Var("c_")
# #     return App(
# #         Function(
# #             (a_, b_, c_),
# #             App(Function((tv0, tv1), IfElse(tv0, Const(0.0), Const(1.0))), (Affine((a_, b_, c_), (x, y)),)),
# #         ),
# #         (a, b, c),
# #     )


# def make_pixel_integral(shader):
#     # Triangle vertices
#     tx1 = Var("tx1")
#     tx2 = Var("tx2")
#     tx3 = Var("tx3")

#     ty1 = Var("ty1")
#     ty2 = Var("ty2")
#     ty3 = Var("ty3")

#     # Pixel boundaries
#     px0 = Var("px0")
#     px1 = Var("px1")

#     py0 = Var("py0")
#     py1 = Var("py1")

#     # Variables of integration.
#     x = TegVar("x")
#     y = TegVar("y")

#     line12_halfspace = half_space(x, y, (ty1 - ty2), (tx2 - tx1), (tx1 * ty2 - tx2 * ty1))
#     line23_halfspace = half_space(x, y, (ty2 - ty3), (tx3 - tx2), (tx2 * ty3 - tx3 * ty2))
#     line31_halfspace = half_space(x, y, (ty3 - ty1), (tx1 - tx3), (tx3 * ty1 - tx1 * ty3))

#     # Build active mask.
#     point_mask = line12_halfspace * line23_halfspace * line31_halfspace

#     alpha = -(x - tx2) * (y - ty3) + (x - tx3) * (y - ty2)
#     beta = -(x - tx3) * (y - ty1) + (x - tx1) * (y - ty3)
#     gamma = -(x - tx1) * (y - ty2) + (x - tx2) * (y - ty1)

#     norm = alpha + beta + gamma
#     mu_x = BoundedLebesgue(0, 1, x)
#     mu_y = BoundedLebesgue(0, 1, y)
#     integral_x = Int(App(shader, (alpha / norm, beta / norm, gamma / norm)) * point_mask, mu_x)
#     integral = Int(integral_x, mu_y)

#     f = Function((tx1, ty1, tx2, ty2, tx3, ty3, px0, px1, py0, py1), integral)
#     return f


# def const_color():
#     tc0, dtc0 = Var("tc0"), Var("dtc0")
#     a, b, c = Var("alpha"), Var("beta"), Var("gamma")
#     return Function((a, b, c), (tc0)), (tc0,), (dtc0,), {tc0.name: dtc0.name}, {tc0.name: 1.0}


# def linear_color():
#     tc0, dtc0 = Var("tc0"), Var("dtc0")
#     tc1, dtc1 = Var("tc1"), Var("dtc1")
#     tc2, dtc2 = Var("tc2"), Var("dtc2")
#     a, b, c = Var("alpha"), Var("beta"), Var("gamma")
#     return (
#         # Function((a, b, c), (tc0 * a + tc1 * b + tc2 * c)),
#         Function((a, b, c), (tc0 * a + tc1 * b + tc2 * c)),
#         (tc0, tc1, tc2),
#         (dtc0, dtc1, dtc2),
#         {tc0.name: dtc0.name, tc1.name: dtc1.name, tc2.name: dtc2.name},
#         {tc0.name: 1.0, tc1.name: 1.0, tc2.name: 1.0},
#     )


# def quadratic_color():
#     tc0, dtc0 = Var("tc0"), Var("dtc0")
#     tc1, dtc1 = Var("tc1"), Var("dtc1")
#     tc2, dtc2 = Var("tc2"), Var("dtc2")
#     tch0, dtch0 = Var("tch0"), Var("dtch0")
#     tch1, dtch1 = Var("tch1"), Var("dtch1")
#     tch2, dtch2 = Var("tch2"), Var("dtch2")

#     a, b, c = Var("alpha"), Var("beta"), Var("gamma")

#     norm = a + b + c

#     a_norm = a
#     b_norm = b
#     c_norm = c

#     n200 = (a_norm) * (2 * a_norm - 1)
#     n020 = (b_norm) * (2 * b_norm - 1)
#     n002 = (c_norm) * (2 * c_norm - 1)

#     n011 = 4 * b_norm * c_norm
#     n101 = 4 * a_norm * c_norm
#     n110 = 4 * a_norm * b_norm

#     color = n200 * tc0 + n020 * tc1 + n002 * tc2 + n011 * tch0 + n101 * tch1 + n110 * tch2

#     return (
#         Function((a, b, c), (color)),
#         (tc0, tc1, tc2, tch0, tch1, tch2),
#         (dtc0, dtc1, dtc2, dtch0, dtch1, dtch2),
#         {
#             tc0.name: dtc0.name,
#             tc1.name: dtc1.name,
#             tc2.name: dtc2.name,
#             tch0.name: dtch0.name,
#             tch1.name: dtch1.name,
#             tch2.name: dtch2.name,
#         },
#         {tc0.name: 0.0, tc1.name: 0.0, tc2.name: 0.0, tch0.name: 0.0, tch1.name: 0.0, tch2.name: 1.0},
#     )


# def test_fragments(color_style):
#     c_shader, c_vars, c_dvars, c_dvar_map, c_value_map = color_style()

#     pixel_value = make_pixel_integral(c_shader)

#     tx1, dtx1 = Var("tx1"), Var("dtx1")
#     tx2, dtx2 = Var("tx2"), Var("dtx2")
#     tx3, dtx3 = Var("tx3"), Var("dtx3")

#     ty1, dty1 = Var("ty1"), Var("dty1")
#     ty2, dty2 = Var("ty2"), Var("dty2")
#     ty3, dty3 = Var("ty3"), Var("dty3")

#     # Pixel boundaries
#     px0, dpx0 = Var("px0"), Var("dpx0")
#     px1, dpx1 = Var("px1"), Var("dpx1")

#     py0, dpy0 = Var("py0"), Var("dpy0")
#     py1, dpy1 = Var("py1"), Var("dpy1")

#     # Triangle color
#     # tc0, dtc0 = Var("tc0"), Var("dtc0")

#     ((vtx1, vty1), (vtx2, vty2), (vtx3, vty3), (vpx0, vpx1), (vpy0, vpy1)) = (
#         (-1, 2),
#         (2, -1),
#         (-2, -2),
#         (0, 1),
#         (0, 1),
#     )
#     values = (vtx1, vty1, vtx2, vty2, vtx3, vty3, vpx0, vpx1, vpy0, vpy1)
#     values = tuple(Const(value) for value in values)

#     dtx1 = Var("dtx1")
#     dexpr = deriv(
#         App(pixel_value, (tx1, ty1, tx2, ty2, tx3, ty3, px0, px1, py0, py1)),
#         context={
#             tx1.name: dtx1.name,
#             ty1.name: dty1.name,
#             tx2.name: dtx2.name,
#             ty2.name: dty2.name,
#             tx3.name: dtx3.name,
#             ty3.name: dty3.name,
#             px0.name: dpx0.name,
#             py0.name: dpy0.name,
#             px1.name: dpx1.name,
#             py1.name: dpy1.name,
#             **c_dvar_map,
#         },
#     )

#     a = evaluate(
#         dexpr,
#         VarVal(
#             {
#                 tx1.name: vtx1,
#                 ty1.name: vty1,
#                 tx2.name: vtx2,
#                 ty2.name: vty2,
#                 tx3.name: vtx3,
#                 ty3.name: vty3,
#                 px0.name: vpx0,
#                 py0.name: vpy0,
#                 px1.name: vpx1,
#                 py1.name: vpy1,
#                 **c_value_map,
#                 dtx1.name: 1.0,
#                 dty1.name: 0.0,
#                 dtx2.name: 1.0,
#                 dty2.name: 0.0,
#                 dtx3.name: 0.0,
#                 dty3.name: 0.0,
#                 dpx0.name: 0.0,
#                 dpy0.name: 0.0,
#                 dpx1.name: 0.0,
#                 dpy1.name: 0.0,
#                 **{dvar.name: 0.0 for dvar in c_dvars},
#             }
#         ),
#     )


# if __name__ == "__main__":
#     test_fragments(const_color)
#     # test_fragments(linear_color)
#     # test_fragments(quadratic_color)
