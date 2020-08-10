import numpy as np


def lerpColor(a, b, k):
    a = np.array(a)
    b = np.array(b)
    return tuple(b*k + a*(1-k))


Map = lambda value, min1, max1, min2, max2: min2 + (max2 - min2) * ((value - min1) / (max1 - min1))


def dist(x1, x2, y1=0, y2=0, z1=0, z2=0):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def Circle(__internal, Window, location, R, boundary, Color, BColor):
    if boundary:
        __internal.circle(Window, BColor, location, R, int(R / 10))
        __internal.circle(Window, Color, location, int(9 * R / 10) + 1)
    else:
        __internal.circle(Window, Color, location, R)


def drawNeat(font, Drawer, Window, Network, Size, Center, boundary=True, font_Name=None, drawText=True,
             LC1=(255, 0, 0), LC2=(0, 0, 255), boundaryColor=(0, 0, 0), txtColor=(0, 0, 0),
             inputNodeColor=(255, 255, 255), outputNodeColor=(255, 255, 255),
             biasNodeColor=(255, 255, 255), hiddenNodeColor=(255, 255, 255)):
    k = Size[0] / Network.layers
    r = [0 for _ in range(Network.layers)]
    for i in Network.nodes:
        r[i.layer] += 1
    N = {}
    R = int(min(Size) / (3 * max(max(r), Network.layers)))
    s = [0 for _ in range(Network.layers)]
    for i in Network.network:
        N[i.number] = (int(k / 2 + i.layer * k - Size[0] / 2 + Center[0]),
                       int((s[i.layer] - r[i.layer] / 2) * 3 * R + 3 * R / 2 + Center[1]))
        s[i.layer] += 1
    MyFont = font.SysFont(font_Name, R)
    for i in Network.network:
        for j in i.output_connections:
            if j.enabled:
                Drawer.line(Window, lerpColor(LC1, LC2, Map(j.w, -1, 1, 0, 1)),
                            N[j.toNode.number], N[i.number], int(abs(Map(j.w, -1, 1, -5, 5))))
        if i.number < Network.input_node:
            Circle(Drawer, Window, N[i.number], R, boundary, inputNodeColor, boundaryColor)
        elif Network.output_node + Network.input_node > i.number >= Network.input_node:
            Circle(Drawer, Window, N[i.number], R, boundary, outputNodeColor, boundaryColor)
        elif i.number == Network.biasNode:
            Circle(Drawer, Window, N[i.number], R, boundary, biasNodeColor, boundaryColor)
        else:
            Circle(Drawer, Window, N[i.number], R, boundary, hiddenNodeColor, boundaryColor)
        if drawText:
            textSurface = MyFont.render(str(i.number), False, txtColor)
            x, y = N[i.number]
            Window.blit(textSurface, (x - textSurface.get_width() / 2, y - textSurface.get_height() / 2))
