import numpy as np
import matplotlib.pyplot as plt
import itertools

import galsim

NX = NY = 64

GAL_HLR = 0.4
PSF_HLR = 0.1

PIXELSCALE = 0.1
INTERPOLANT = "lanczos11" 

DGAMMA = 0.01

SHEARS = {
    "NOSHEAR": galsim.Shear(g1=0, g2=0),
    "1P": galsim.Shear(g1=DGAMMA, g2=0),
    "1M": galsim.Shear(g1=-DGAMMA, g2=0),
    "2P": galsim.Shear(g1=0, g2=DGAMMA),
    "2M": galsim.Shear(g1=0, g2=-DGAMMA),
}

def get_background(a=0.0, b=0.0, c=30):
    x, y = np.meshgrid(np.linspace(-1, 1, NX), np.linspace(-1, 1, NY))

    return a*x + b*y + c

def moments(stamp, sigw=5, maxits=50, prec=1E-3):

    def gaussian(xc, yc, sigw):
        x = np.arange(NX)
        y = np.arange(NY)

        gx = np.exp(-(x-xc)**2 / sigw**2)
        gy = np.exp(-(y-yc)**2 / sigw**2)

        return np.outer(gy, gx)

    xc = NX // 2
    yc = NY // 2
    x = np.arange(NX)
    y = np.arange(NY)

    # centroiding
    for _ in range(maxits):
        w = gaussian(xc, yc, sigw)

        sw = stamp * w
        sx = sw.sum(0)
        sy = sw.sum(1)
        s = sw.sum()

        dx = (sx*(x-xc)).sum() / s
        dy = (sy*(y-yc)).sum() / s

        xc += dx
        yc += dy

        if np.hypot(dx, dy) < prec:
            break

    else:
        print(f"Warning... centroid finding did not converge after {maxits} iterations")

    w = gaussian(xc, yc, sigw)
    sw = stamp * w
    sx = sw.sum(0)
    sy = sw.sum(1)
    s = sw.sum()

    qxx = (sx*(x-xc)**2).sum() / s
    qyy = (sy*(y-yc)**2).sum() / s
    qxy = (sw * np.outer(y-yc, x-xc)).sum() / s

    e1 = (qxx-qyy) / (qxx+qyy)
    e2 = 2*qxy / (qxx+qyy)
    r2 = qxx+qyy

    return galsim.Shear(e1=e1, e2=e2), r2

def make_galaxy_images(bkg):
    gal = galsim.Sersic(n=1, half_light_radius=GAL_HLR, flux=1000)
    psf = galsim.Gaussian(half_light_radius=PSF_HLR, flux=1)

    if np.sum(bkg**2) > 0:

        bkg_galsim = galsim.InterpolatedImage(
            galsim.Image(bkg, scale=PIXELSCALE),
            x_interpolant=INTERPOLANT,
            k_interpolant=INTERPOLANT,
        )

        gal += bkg_galsim

    imgs = {}

    for label, shear in SHEARS.items():

        sheared_gal = gal.shear(shear)
        reconv_gal = galsim.Convolve(sheared_gal, psf)

        img = reconv_gal.drawImage(nx=NX, ny=NY, scale=PIXELSCALE).array
        imgs[label] = img

    return imgs

def measure(gal_images, sigw=5):

    mes = {}

    for label, img in gal_images.items():
        s, r2 = moments(img, sigw=sigw)
        mes[label] = s

    g1 = mes["NOSHEAR"].g1
    g2 = mes["NOSHEAR"].g2

    r11 = (mes["1P"].g1 - mes["1M"].g1) / (2*DGAMMA)
    r12 = (mes["1P"].g2 - mes["1M"].g2) / (2*DGAMMA)
    r21 = (mes["2P"].g1 - mes["2M"].g1) / (2*DGAMMA)
    r22 = (mes["2P"].g2 - mes["2M"].g2) / (2*DGAMMA)

    print("Response", (r11 + r22)/2)

if __name__ == "__main__":

    for sigw in (2, 5, 10):

        print("SIGW=", sigw)

        # correct background
        bkg = get_background(0, 0, 0)
        gals = make_galaxy_images(bkg)
        print("Correct BKG")
        measure(gals, sigw)

        # wrong background (+1 ADU)
        print("Background 1 ADU too faint")
        bkg = get_background(0, 0, 1)
        gals = make_galaxy_images(bkg)
        measure(gals, sigw)

        # wrong background (-1 ADU)
        print("Background 1 ADU too bright")
        bkg = get_background(0, 0, -1)
        gals = make_galaxy_images(bkg)
        measure(gals, sigw)

        # wrong background (x gradent)
        print("Background gradient in x")
        bkg = get_background(1, 0, 0)
        gals = make_galaxy_images(bkg)
        measure(gals, sigw)

        # wrong background (y gradent)
        print("Background gradient in y")
        bkg = get_background(0, 1, 0)
        gals = make_galaxy_images(bkg)
        measure(gals, sigw)

        # wrong background (xy gradent)
        print("Background gradient in xy")
        bkg = get_background(0.5, 0.5, 0)
        gals = make_galaxy_images(bkg)
        measure(gals, sigw)