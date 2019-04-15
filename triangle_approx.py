def cntr_approx(points ,step, th):
    contour = points.copy()
    approx = []
    i = 0
    f = step
    length = len(contour)
    start_point = contour[i]
    ff = i + 2 * f
    end_point = contour[ff]
    B = contour[i + f]
    garbage = (0, 0)
    for k in range(2 * ff):
        contour.append(garbage)
    while True:
        print(i)
        base = dst(end_point, start_point)
        height = triangle_sqr(start_point, B, end_point) / base
        if height / base >= th:
            i = ff - f
            start_point = B
            B = end_point
            ff += 2 * f
            end_point = contour[ff]
        else:
            approx.append(start_point)
            tmp_ff = ff
            ff += 2 * f
            i = ff
            start_point = end_point
            B = contour[tmp_ff + f]
            end_point = contour[i]
        if end_point == (0, 0):
            break
    return approx
