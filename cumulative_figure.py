import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as legend


def cumulative_rank(x):
    '''
    ==========================================================
    Demo of using histograms to plot a cumulative distribution
    ==========================================================

    This shows how to plot a cumulative, normalized histogram as a
    step function in order to visualize the empirical cumulative
    distribution function (CDF) of a sample. We also use the ``mlab``
    module to show the theoretical CDF.

    A couple of other options to the ``hist`` function are demonstrated.
    Namely, we use the ``normed`` parameter to normalize the histogram and
    a couple of different options to the ``cumulative`` parameter.
    The ``normed`` parameter takes a boolean value. When ``True``, the bin
    heights are scaled such that the total area of the histogram is 1. The
    ``cumulative`` kwarg is a little more nuanced. Like ``normed``, you
    can pass it True or False, but you can also pass it -1 to reverse the
    distribution.

    Since we're showing a normalized and cumulative histogram, these curves
    are effectively the cumulative distribution functions (CDFs) of the
    samples. In engineering, empirical CDFs are sometimes called
    "non-exceedance" curves. In other words, you can look at the
    y-value for a given-x-value to get the probability of and observation
    from the sample not exceeding that x-value. For example, the value of
    225 on the x-axis corresponds to about 0.85 on the y-axis, so there's an
    85% chance that an observation in the sample does not exceed 225.
    Conversely, setting, ``cumulative`` to -1 as is done in the
    last series for this example, creates a "exceedance" curve.

    Selecting different bin counts and sizes can significantly affect the
    shape of a histogram. The Astropy docs have a great section on how to
    select these parameters:
    http://docs.astropy.org/en/stable/visualization/histogram.html
    =======================================================================
    ********* 이사람 코드 받아서 Re-ID 의 표출에 맞게 수정함 *************
    =======================================================================
    '''

    max_rank = 50  # display 할 최대 rank

    y = x/x.size  # [0 1] 크기
    y_dis = y[0:50]  # start offset 부터 end-1 offset 까지

    x_bin = list(range(0, max_rank))

    plt.figure(99)
    plt.plot(x_bin, y_dis, 'r-')  # (x 축, y축, '옵션')

    print('************ rank accuracy ************')
    print(y*100, '%')

    # plt.legend([acc], ['Re-ID curve'], loc=1)
    plt.title('CMC curves (Re_ID)')
    plt.xlabel('rank')
    plt.ylabel('identification rate')
    plt.axis([-0.05, max_rank+2, 0, 1.05])  # x축 표현 범위, y축 표현범위
    plt.grid(True)
    plt.show()

    print('그래프 와 데이터 보게 stop')