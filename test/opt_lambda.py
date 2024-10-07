import math
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt


def main():
    N_term_est = 20
    alpha = 1 - 1 / N_term_est  # 用折扣系数 gamma 给出压缩系数上界
    n_lam_intervs = 10000
    horizon_TDlambda = np.clip(N_term_est, 1, 50)
    rst_opt = []
    for i_m, m in enumerate(range(1, horizon_TDlambda + 1)):
        # min_{\lambda} C(\lambda,m,\alpha)
        rst = []
        for i_lam in range(n_lam_intervs):
            lam = i_lam / n_lam_intervs
            a_l = alpha * lam
            y = (1 - lam) * ((1 - a_l**m) / (1 - a_l)) + lam**m
            rst.append((lam, y))

        i_lam = np.argmin(np.array(rst)[:, 1])
        lam_opt = rst[i_lam][0]
        C_opt = rst[i_lam][1]
        print(f"m={m}, lambda_*={lam_opt:.04f}, C_*={C_opt:.04f}")

        ave_contrib = (1 - C_opt) / m
        par_contrib = (1 - C_opt) if i_m == 0 else (rst_opt[i_m - 1][2] - C_opt)
        eb_ratio = abs(1 - C_opt / (alpha**m))
        cmplxty = math.log(alpha * C_opt) / m

        rst_opt.append(
            (
                m,
                lam_opt,
                C_opt,
                ave_contrib,
                par_contrib,
                eb_ratio,
                cmplxty,
            )
        )

    rst_opt = np.asarray(rst_opt)

    fig = plt.figure()
    ncols_plt = 2
    meta = [
        (rst_opt[:, 1], r"\lambda_*", 0),
        (rst_opt[:, 2], r"C_*", -1),
        (rst_opt[:, 3], r"\frac{1-C_*}{m}", 1),
        (rst_opt[:, 4], r"\Delta C_*", 1),
        (rst_opt[:, 5], r"\left|\frac{C_*}{\alpha^m}-1\right|", -1),
        (rst_opt[:, 6], r"\log(\alpha C_*) / m", -1),
    ]
    nrows_plt = int(np.ceil(len(meta) / ncols_plt))
    ms = rst_opt[:, 0]
    ls = rst_opt[:, 1]
    for i_meta, (ys, label, search_direction) in enumerate(meta):
        ax = fig.add_subplot(nrows_plt, ncols_plt, i_meta + 1)
        ax.plot(ms, ys, ".-", label=f"${label}$")
        if search_direction:
            tag = "max" if search_direction > 0 else "min"
            i_opt = np.argmax(ys) if search_direction > 0 else np.argmin(ys)
            ax.plot(ms[i_opt], ys[i_opt], "*", label=rf"$\{tag}_m {label}$")
            tt = ", ".join(
                [
                    rf"\alpha={alpha:.04f}",
                    rf"m_*={int(ms[i_opt])}",
                    rf"\lambda_*={float(ls[i_opt]):.04f}",
                    rf"{ys[i_opt]:.04f}",
                ]
            )
            ax.set_title(f"${tt}$")
        ax.set_xlabel(r"$m$")
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
