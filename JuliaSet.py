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
    max_abs = 4
    for _ in range(level):
        norm = np.abs(Z)
        bb = norm < max_abs
        Z[bb] = Z[bb] * Z[bb] + c
        Z_level[bb] -= 4
        Z_level[Z_level < 0] = 0
    return Z, Z_level


def list_of_colormaps():
    # http://matplotlib.org/examples/color/colormaps_reference.html
    cmaps = [('Perceptually Uniform Sequential', [
                'viridis', 'plasma', 'inferno', 'magma']),
            ('Sequential', [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
            ('Sequential (2)', [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper']),
            ('Diverging', [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
            ('Qualitative', [
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                'tab10', 'tab20', 'tab20b', 'tab20c']),
            ('Miscellaneous', [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
    all_cmaps = [cmap_list for _, cmap_list in cmaps]
    flatten = lambda l: [item for sublist in l for item in sublist]
    all_cmaps = flatten(all_cmaps)
    return all_cmaps


def make_image(data, outputname='res.png', **kwargs):
    import matplotlib.pyplot as plt
    import os
    dOutputName = os.path.dirname(outputname)
    if dOutputName and not os.path.exists(dOutputName):
        os.makedirs(dOutputName)
    xmin = kwargs.get('xmin', 0.0)
    ymin = kwargs.get('ymin', 0.0)
    xmax = kwargs.get('xmax', 1.0)
    ymax = kwargs.get('ymax', 1.0)
    dpi = kwargs.get('dpi', 40)
    colormap = kwargs.get('colormap', 'hot')
    fig = plt.figure()
    fig.set_size_inches(1*(xmax-xmin), 1*(ymax-ymin))
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap(colormap)
    # ax.contourf(data, aspect='normal')
    ax.contourf(data)
    plt.savefig(outputname, dpi=dpi)
    plt.close()


def getFilename(colormap, c, suffix=''):
    return colormap + '/res_{0:06d}_{1:06d}{2}.png'.format(int(100000 * c.real), int(100000 * c.imag), suffix)


def create_one_julias_set(c=complex(0.0, 0.65), colormap='magma', outputname=None, **kwargs):
    s = kwargs.get('s', 401)
    x = kwargs.get('x', 2.0)
    invert = kwargs.get('invert', False)
    Z, Z_level = julia_set(w=s, h=s, c=c, re_min=-x, re_max=+x, im_min=-x, im_max=+x, **kwargs)
    if outputname is None:
        outputname = getFilename(colormap, c, suffix='')
    make_three_images = False
    if make_three_images:
        ZZ = np.concatenate((np.real(Z), np.abs(Z), np.imag(Z)), axis=1)
        make_image(ZZ, outputname=outputname, xmax=3, colormap=colormap)
    else:
        ZZ = np.abs(Z)
        make_image(ZZ, outputname=outputname, colormap=colormap, dpi=s)
    if invert:
        outputname = getFilename(colormap, c, '_level')
        make_image(Z_level, outputname=outputname, colormap=colormap, dpi=s)
        outputname = getFilename(colormap, c, '_level_mix')
        make_image(Z_level+np.abs(Z), outputname=outputname, colormap=colormap, dpi=s)


def create_several_julias_set(n, cn, **kwargs):
    parallel = kwargs.get('parallel', False)
    cn = np.linspace(cn[0], cn[1], n)
    if parallel:
        from multiprocessing import cpu_count
        from multiprocessing import Pool
        ncores = max(1, cpu_count()-1)
        raise NotImplementedError
        listOfInputs = [{'base':b, 'factor':f, 'text':args.text, 'filename':k, 'dpi':args.dpi} for f, b, k in zip(factors, bases, filenames)]
        p = Pool(ncores)
        p.map(create_one_julias_set, listOfInputs)
    else:
        for c in cn:
            create_one_julias_set(c, **kwargs)


def create_animated_gif(maxRecursionLevel=6, filename='juliaset.gif', **kwargs):
    grid = kwargs.get('grid', False)
    import subprocess
    generateLevel = lambda x: list(range(x)) + [x - i - 2 for i in range(x - 1)]
    cmd = 'convert -antialias -density 100 -delay 120 '
    for level in generateLevel(maxRecursionLevel + 1):
        cfilename = filename + '_' + '{0:03d}'.format(level) + '.png'
        cmd += cfilename + ' '
        plot_level(max_level=level, showAllLevel=False, filename=cfilename)
    cmd += filename
    subprocess.check_output(cmd.split(' '))


def create_animated_gif2(filename='juliaset.gif', **kwargs):
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
    subprocess.check_output(cmd.split(' '))
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

        #
        python JuliaSet.py -i -s 501 -x 1.25 -k 0.285+0.01j

        #
        python JuliaSet.py -i -s 501 -x 1.25 -k 0.285+0.01j 0.285+0.02j -n 50

        # Same as previous but with fully developped argument
        python JuliaSet.py --invert --size 501 -x 1.25 -k 0.285+0.01j 0.285+0.02j -number 50
        """
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
    pa('-s', '--size', type=int, help='number of recursion level. Reasonnable value is 6', default=401)
    pa('-c', '--colormap', type=str, help='number of recursion level. Reasonnable value is 6', default='viridis')
    pa('-i', '--invert', action='store_true', help='boolean used to display all levels')
    pa('-o', '--output', default=None, help='name of the generated file. If not provided, result will display on screen')
    pa('-x', type=float, default=2.0, help='domain size of the fractal. Default is 2.0, meaning a -2 x +2, -2 x +2 square will be created')
    pa('-k', type=complex, default=[complex(0.285, 0.01)], nargs='*', help='name of the generated file. If not provided, result will display on screen')
    pa('-n', '--number', type=int, help='number of pictures to generate between two complex numbers. Default is 2', default=2)
    pa('-p', '--parallel', action='store_true', help='boolean used to create images in a parallel way. It used the (n-1) cores. Default is False')
    args = parser.parse_args()
    output = args.output
    if len(args.k) == 1:
        create_one_julias_set(c=args.k[0], colormap=args.colormap, outputname=output, s=args.size, x=args.x, invert=args.invert)
    else:
        create_several_julias_set(n=args.number, cn=args.k, colormap=args.colormap, outputname=output, s=args.size, x=args.x, invert=args.invert)
        if args.output is None:
            args.output = 'juliaset.mp4'
        if args.output and args.output.lower().endswith('gif'):
            raise NotImplementedError
            # create_animated_gif
        elif args.output and args.output.lower().endswith('mp4'):
            raise NotImplementedError
            # create_animated_mp4
        else:
            raise Exception('Invalid extension, one expects gif or mp4')

if __name__ == '__main__':
    main()
