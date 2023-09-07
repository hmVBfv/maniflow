import numpy as np


def mixedColor(imageBuffer: np.array, color: np.array) -> np.array:
    """
    This method implements subtractive color mixing for RGBA colors (or rather it
    is a heuristic to approach the problem since there are no physical models
    that accurately implement this)
    RGBA naturally supports additive color mixing but for translucent colored layers that we want to
    stack on top of each other we want subtractive color mixing - the colors should get darker
    each time layers are stacked. Or to put it in other words: if faces overlap in the resulting
    rendered image we want their colors to mix so that the resulting color is darker than the colors we started
    with. The opacity should be the sum of both of the opacities of the faces (constrained to 255).

    :param imageBuffer: the color of the image buffer at some pixel in the rendered image
    :param color: the color of a (new) polygon that is to be added to the rendered scene
    :return: the mixed color of both input colors,
    """
    color = np.array(color)
    color[:3:] = 255 * np.ones(3) - color[:3:]  # we first 'invert' the colorB
    result = np.array([0, 0, 0, 0])
    result[:3:] = np.sqrt((imageBuffer[:3:] ** 2 + color[:3:] ** 2) / 2)  # we sort of mix the colors
    # not just by their average but according to some other formula I came up with that seemed to work nicely...
    # this formula 'blends' the colors better
    result[3] = imageBuffer[3] + color[3]  # we add the opacities of the colors
    # now we apply the constraints
    result[result > 255] = 255

    return result


def baryCentricCoordinates(a: np.array, b: np.array, c: np.array, x: int, y: int) -> np.array:
    """
    A method to compute the barycentric coordinates of a point (x,y) with respect to
    the triangle that is given by the three corners a, b and c.
    :param a: a corner of the triangle
    :param b: a corner of the triangle
    :param c: a corner of the triangle
    :param x: the x component of the cartesian coordinates of the points
    :param y: the y component of the cartesian coordinates of the points
    :return: the barycentric coordinates of the given points with respect to the triangle
    """

    point = np.array([x, y])  # representing the point as a numpy array
    edge1 = b - a  # computing to edges of the triangle
    edge2 = c - a
    p = point - a  # shifting the whole coordinate system so that a is the origin

    area = np.cross(edge2, edge1)  # the determinant of the matrix where edge2 and edge1 are column vectors
    beta = np.cross(edge2, p) / area  # corresponding areas of the partial triangles
    gamma = np.cross(p, edge1) / area  # The partial triangles are obtained by drawing the line
    # from each corner point onto the point.
    return np.array([1 - beta - gamma, beta, gamma])


def getBoundingBox(a: np.array, b: np.array, c: np.array) -> list[list[float]]:
    """
    Returns two corner points that 'span' a bounding box (rectangle) around a given triangle
    with corner points a, b and c. The bounding box is the smallest rectangle that
    encloses the whole triangle.

    This method returns a list that consists of two lists.
    The first list consists of two floats whereby the first float
    is the absolute smallest x-component of all the corner points. The
    second float is the smallest y-component of all the corner points.
    The second list is constructed analogously (we just choose the largest component
    each time)
    :param a: a corner point of the triangle
    :param b: a corner point of the triangle
    :param c: a corner point of the triangle
    :return: a list that consists of the coordinates of two
    points that 'span' the bounding box.
    """

    return [[min([xx[0] for xx in [a, b, c]]), min([yy[1] for yy in [a, b, c]])],
            [max([xx[0] for xx in [a, b, c]]), max([yy[1] for yy in [a, b, c]])]]


def rasterizeLine(x0: int, y0: int, x1: int, y1: int, color: np.array, img: np.array) -> np.array:
    """
    This method implements Bresenham's line algorithm
    (see: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)
    The implementation is basically taken from the wikipedia page
    (but of course translated into python).

    Bresenham's line algorithm is an algorithm used for drawing lines from the point (x0,y0)
    to (x1,y1) on a raster.
    The algorithm finds a close approximation of the line and maps it to the image buffer (img).
    :param x0: the x component of the starting point of the line
    :param y0: the y component of the starting point of the line
    :param x1: the x component of the end point of the line
    :param y1: the y component of the end point of the line
    :param color: the color (RGBA) with which the line is to be drawn
    :param img: the image buffer
    :return: the image buffer with the line rendered
    """

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    while True:
        try:
            img[x0][y0] = mixedColor(img[x0][y0], color)
        except Exception as _:
            pass
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                break
            error += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            error += dx
            y0 += sy
    return img


def rasterizeTriangle(a: np.array, b: np.array, c: np.array, color: np.array, img: np.array) -> np.array:
    """
    This method rasterizes a given triangle with corner points a, b and c
    onto the image buffer (img).

    First, the bounding box is computed. We then scan every point in the bounding box
    and check whether it lies within the triangle. The point is in the triangle
    iff all components of the barycentric coordinates of that point with respect to the
    triangle are positive.

    :param a: a corner point of the triangle
    :param b: a corner point of the triangle
    :param c: a corner point of the triangle
    :param color: an RGBA color represented by an array consisting of four bytes
    :param img: the image buffer
    :return: the image buffer with the rasterized triangle
    """

    box = getBoundingBox(a, b, c)
    for x in range(int(np.floor(box[0][0])), int(np.ceil(box[1][0]))):  # we scan every point (x,y) in the bounding box
        for y in range(int(np.floor(box[0][1])), int(np.ceil(box[1][1]))):
            baryCentric = baryCentricCoordinates(a, b, c, x, y)  # we compute the barycentric coordinates
            inTriangle = np.all(baryCentric >= 0)
            # now we check whether this point is inside the image buffer (if not we discard it)
            if not inTriangle or not 0 <= x < img.shape[0] or not 0 <= y < img.shape[1]:
                continue

            # if the point lies within the image buffer and in the triangle we apply the color to it
            img[x][y] = mixedColor(img[x][y], color)

    return img


def rasterizePolygon(polygon: list[np.array], image: np.array, fill: np.array = None,
                     opacity: int = 255, stroke: np.array = None, *args, **kwargs):
    """
    A method to either rasterize a triangle or a quadrilateral.
    The quadrilateral is broken down into two triangles.

    :param polygon: a list of points that make up the corners of the polygon
    :param image: the image buffer
    :param fill: an RGB color with which the polygon is to be painted
    :param opacity: the opacity for the rasterized polygon
    :param stroke: the RGBA color with which the edge of the polygon should be painted
    :return: the image buffer with the rendered polygon
    """

    if fill is not None and opacity != 0:
        fill = list(fill)
        fill.append(opacity)  # we construct the RGBA color from the specified values of fill and opacity
        if len(polygon) == 3:
            # in this case we simply rasterize a triangle
            image = rasterizeTriangle(polygon[0], polygon[1], polygon[2], color=fill, img=image)
        if len(polygon) == 4:
            # in this case we rasterize two triangles that make up the quadrilateral
            image = rasterizeTriangle(polygon[0], polygon[1], polygon[2], color=fill, img=image)
            image = rasterizeTriangle(polygon[0], polygon[3], polygon[2], color=fill, img=image)

    if stroke is None:
        return image  # if there is no stroke color specified we are done

    # we now rasterize the edges of the polygon using Bresenham's line algorithm
    pp = polygon[0]
    for p in polygon[1::]:
        image = rasterizeLine(int(pp[0]), int(pp[1]), int(p[0]), int(p[1]), color=stroke, img=image)
        pp = p
    image = rasterizeLine(int(polygon[0][0]), int(polygon[0][1]), int(polygon[-1][0]),
                          int(polygon[-1][1]), color=stroke, img=image)

    return image
