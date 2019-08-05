#!/usr/bin/env python3
#
# This script creates ...
#
# Requires a Python interpreter with numpy and matplotlib
#
# Requires imagemagick to access convert when creating gif
# Requires ffmpeg to create mp4 video
#
# Examples
# python JuliaSet.py --help
# python JuliaSet.py -i -s 501 -x 1.25 -k 0.285+0.01j
# python JuliaSet.py -i -s 501 -x 1.25 -k 0.285+0.01j 0.285+0.02j -n 50
#
# https://trac.ffmpeg.org/wiki/Slideshow
# cat *.png | ffmpeg -f image2pipe -i - output.mp4

import numpy as np

def julia_set(**kwargs):
    # Specify image width and height
    w = kwargs.get('w', 401)
    h = kwargs.get('h', 401)
    c = kwargs.get('c', complex(0.0, 0.65))
    level = kwargs.get('level', 255)
    # Specify real and imaginary range of image
    re_min = kwargs.get('re_min', -2.0)
    re_max = kwargs.get('re_max', +2.0)
    im_min = kwargs.get('im_min', -2.0)
    im_max = kwargs.get('im_max', +2.0)

    # Generate evenly spaced values over real and imaginary ranges
    real_range = np.linspace(re_min, re_max, w)
    imag_range = np.linspace(im_min, im_max, h)
    Y, X = np.meshgrid(imag_range, real_range)
    Z = X + 1j * Y
    Z_level = level * np.ones(Z.shape, dtype=np.uint8)
    max_abs = 6
    for _ in range(level):
        norm = np.abs(Z)
        bb = norm < max_abs
        Z[bb] = Z[bb] * Z[bb] + c
        Z_level[bb] -= 5
        Z_level[Z_level < 0] = 0
    return Z, Z_level


def list_of_colormaps():
    # http://matplotlib.org/examples/color/colormaps_reference.html
    import matplotlib.pyplot as plt
    # Get a list of the colormaps in matplotlib.  Ignore the ones that end with
    # '_r' because these are simply reversed versions of ones that don't end
    # with '_r'
    return sorted(m for m in plt.cm.datad if not m.endswith("_r"))


def make_image(data, outputname='res.png', **kwargs):
    import matplotlib.pyplot as plt
    import os
    if outputname:
        dOutputName = os.path.dirname(outputname)
        if dOutputName and not os.path.exists(dOutputName):
            os.makedirs(dOutputName)
    xmin = kwargs.get('xmin', 0.0)
    ymin = kwargs.get('ymin', 0.0)
    xmax = kwargs.get('xmax', 1.0)
    ymax = kwargs.get('ymax', 1.0)
    dpi = kwargs.get('dpi', 40)
    colormap = kwargs.get('colormap', 'hot')
    if colormap == 'all':
        for c in list_of_colormaps():
            fig = plt.figure()
            fig.set_size_inches(1 * (xmax - xmin), 1 * (ymax - ymin))
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.set_cmap(c)
            ax.contourf(data, aspect='normal')
            # ax.contourf(data)
            if outputname:
                name = os.path.splitext(outputname)[0] + '_' + c
                ext = os.path.splitext(outputname)[1]
                plt.savefig(name + ext, dpi=dpi)
                plt.close()
            else:
                plt.savefig(c + '.png', dpi=dpi)
                plt.close()
    else:
        fig = plt.figure()
        fig.set_size_inches(1 * (xmax - xmin), 1 * (ymax - ymin))
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.set_cmap(colormap)
        ax.contourf(data, aspect='normal')
        # ax.contourf(data)
        if outputname:
            plt.savefig(outputname, dpi=dpi)
            plt.close()
        else:
            plt.show()


def get_filename(colormap, c, suffix=''):
    return colormap + '/res_{0:06d}_{1:06d}{2}.png'.format(int(100000 * c.real), int(100000 * c.imag), suffix)


def get_filenames(colormap, cn, suffix=''):
    return [get_filename(colormap, c, suffix) for c in cn]


def create_one_julias_set(c=complex(0.0, 0.65), colormap='magma', outputname=None, **kwargs):
    s = kwargs.get('s', 401)
    x = kwargs.get('x', 2.0)
    w = kwargs.get('w', 0.75)  # define ponderation between level and norm
    pattern = kwargs.get('pattern', 'abcde')  #
    Z, Z_level = julia_set(w=s, h=s, c=c, re_min=-x, re_max=+x, im_min=-x, im_max=+x, **kwargs)
    Z /= np.max(np.abs(Z))
    Z_abs = np.abs(Z)
    Z_level = Z_level.astype(float)
    Z_level = (Z_level - np.min(Z_level)) / (np.max(Z_level) - np.min(Z_level))
    Z_weighted = 0.5 / (w + 1.0) * (Z_level + w * Z_abs)
    Z_weighted = (Z_weighted - np.min(Z_weighted)) / (np.max(Z_weighted) - np.min(Z_weighted))
    dictPattern = {'a': np.real(Z), 'b': Z_abs, 'c': np.imag(Z), 'd': Z_level, 'e': Z_weighted}
    ZZZ = np.concatenate([dictPattern[p] for p in pattern], axis=1)
    make_image(ZZZ, outputname=outputname, xmax=len(pattern), colormap=colormap, dpi=s)


def create_one_julias_set_to_expand(kwargs):
    create_one_julias_set(**kwargs)


def create_several_julias_set(n, cn, **kwargs):
    parallel = kwargs.get('parallel', False)
    colormap = kwargs.get('colormap', 'magma')
    s = kwargs.get('s', 401)
    x = kwargs.get('x', 2.0)
    pattern = kwargs.get('pattern', 'abcde')
    cn = np.linspace(cn[0], cn[1], n)
    outputnames = get_filenames(colormap, cn)
    if parallel:
        from multiprocessing import cpu_count
        from multiprocessing import Pool
        ncores = max(1, cpu_count() - 1)
        listOfInputs = [{'c':complex(c), 's':s, 'x':x, 'pattern':pattern, 'colormap':colormap, 'outputname':outputname} for c, outputname in zip(cn, outputnames)]
        p = Pool(ncores)
        p.map(create_one_julias_set_to_expand, listOfInputs)
    else:
        for c, outputname in zip(cn, outputnames):
            create_one_julias_set(c, outputname=outputname, **kwargs)


def create_animated_gif(filename='juliaset.gif', **kwargs):
    import subprocess
    pngs = kwargs.get('pngs', None)
    continuous = kwargs.get('continuous', False)
    if pngs is None:
        from glob import glob
        pngs = glob('*.png')
    if continuous:
        pngs += pngs[-2:0:-1]
    cmd = 'convert -antialias -density 100 -delay 120 '
    cmd += ' '.join(pngs)
    cmd += ' ' + filename
    subprocess.check_output(cmd.split(' '))


def create_animated_mp4(filename='juliaset.mp4', **kwargs):
    import subprocess
    import os
    pngs = kwargs.get('pngs', None)
    framerate = kwargs.get('framerate', 12)
    continuous = kwargs.get('continuous', False)
    if pngs is None:
        from glob import glob
        pngs = glob('*.png')
    if continuous:
        pngs += pngs[-2:0:-1]
    infile = open('tmp.txt', 'w')
    for png in pngs:
        infile.write('file ' + png + '\n')
        infile.write('duration ' + str(1.0 / framerate) + '\n')
    infile.write('file ' + pngs[-1] + '\n')
    infile.close()
    cmd = 'ffmpeg -f concat -i tmp.txt ' + filename
    subprocess.check_output(cmd.split(' '), cwd=os.getcwd())
    os.remove('tmp.txt')


def get_description():
    from textwrap import dedent
    description = """
        Generate a julias set fractal curve
        """
    return dedent(description)


def get_epilog():
    from textwrap import dedent
    epilog = """
        # Display help
        python JuliaSet.py --help

        # Create a Julia set fractal with k=0.285+0.01j with 501 points and a square with half-length of 1.25
        python JuliaSet.py -s 501 -x 1.25 -k 0.285+0.01j

        # Create an animation with 50 frames that covers from 0.285+0.01j to 0.285+0.02j
        python JuliaSet.py -s 501 -x 1.25 -k 0.285+0.01j 0.285+0.02j -n 50 -o animation.mp4

        # Same as previous but with fully developped argument and parallel execution
        python JuliaSet.py --size 501 -x 1.25 -k 0.285+0.01j 0.285+0.02j -number 50 --parallel

        # Pattern describes the output displayed. Possible letter are:
        a : Real part of Z
        b : Norm of Z
        c : Imaginary part of Z
        d : Convergence level
        e : Ponderation between norm and convergence level

        # Available colormaps are
        {0}
        """.format(', '.join(list_of_colormaps()))
    return dedent(epilog)


def main():
    import argparse
    import os
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(description=get_description(),
                                     epilog=get_epilog(),
                                     formatter_class=CustomFormatter)
    pa = parser.add_argument
    pa('-k', type=complex, default=[complex(0.285, 0.01)], nargs='*', help='complex number used to create Julia set. Two numbers have to be given for animation. Default is 0.285+0.01j.')
    pa('-s', '--size', type=int, help='size of the generated image.', default=401)
    pa('-x', type=float, default=2.0, help='domain size of the fractal. Default is 2.0, meaning a -2 x +2, -2 x +2 square will be created.')
    pa('-c', '--colormap', type=str, help='name of the matplotlib colormap to use. Using "all" creates images for all colormaps', default='autumn')
    pa('--pattern', type=str, help='pattern use to control output. A letter corresponds to a picture, several letters can be given. Description is given below.', default='abcde')
    pa('-o', '--output', default=None, help='name of the generated file. If not provided, result will display on screen.')
    pa('-n', '--number', type=int, help='number of pictures to generate between two complex numbers. Default is 2.', default=2)
    pa('-p', '--parallel', action='store_true', help='boolean used to create images in a parallel way. It used the (n-1) cores. Default is False.')
    args = parser.parse_args()
    output = args.output
    if len(args.k) == 1:
        create_one_julias_set(c=args.k[0], colormap=args.colormap, outputname=output, s=args.size, x=args.x, pattern=args.pattern)
    else:
        create_several_julias_set(n=args.number, cn=args.k, colormap=args.colormap,
              s=args.size, x=args.x, pattern=args.pattern, parallel=args.parallel)
        if args.output is None:
            args.output = 'juliaset.mp4'
        outputnames = get_filenames(cn=np.linspace(args.k[0], args.k[1], args.number),
                                    colormap=args.colormap)
        if args.output.lower().endswith('gif'):
            create_animated_gif(filename=args.output, pngs=outputnames, continuous=True)
        elif args.output.lower().endswith('mp4'):
            create_animated_mp4(filename=args.output, pngs=outputnames, continuous=True)
        else:
            raise Exception('Invalid extension, one expects gif or mp4')

if __name__ == '__main__':
    main()
