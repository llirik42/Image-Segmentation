import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import opendatasets as od


def add_nodes(harris_result: np.ndarray, threshold: int, graph: nx.Graph) -> None:
    height, width = harris_result.shape

    for y in range(height):
        for x in range(width):
            r_value: np.float32 = harris_result[y, x]
            if r_value > threshold:
                graph.add_node((x, y))


def add_edges(mask: np.ndarray, threshold: int, graph: nx.Graph) -> None:
    nodes: list[tuple[int, int]] = list(graph.nodes)

    for i in range(len(nodes)):
        n1: tuple[int, int] = nodes[i]
        x1: int = n1[0]
        y1: int = n1[1]
        for j in range(i + 1, len(nodes)):
            n2: tuple[int, int] = nodes[j]
            x2: int = n2[0]
            y2: int = n2[1]
            min_x: int = min(x1, x2)
            max_x: int = max(x1, x2)
            min_y: int = min(y1, y2)
            max_y: int = max(y1, y2)
            intensity: float = mask[min_y:max_y, min_x:max_x].sum()
            size1: int = max_x - min_x + 1
            size2: int = max_y - min_y + 1
            area: int = size1 * size2
            intensity /= area
            if intensity > threshold:
                graph.add_edge(n1, n2)


def draw_nodes(graph: nx.Graph, mask_nodes_image: np.ndarray, graph_image: np.ndarray) -> None:
    color: tuple[int, int, int] = (255, 0, 0)
    thickness: int = 10
    radius: int = 10

    for x, y in graph.nodes:
        cv2.circle(img=mask_nodes_image, center=(x, y), radius=radius, color=color, thickness=thickness)
        cv2.circle(img=graph_image, center=(x, y), radius=radius, color=color, thickness=thickness)


def draw_edges(graph: nx.Graph, graph_image: np.ndarray) -> None:
    color: tuple[int, int, int] = (0, 255, 0)
    thickness: int = 3

    for edge in graph.edges:
        n1 = edge[0]
        n2 = edge[1]
        x1 = n1[0]
        y1 = n1[1]
        x2 = n2[0]
        y2 = n2[1]
        cv2.line(graph_image, (x1, y1), (x2, y2), color=color, thickness=thickness)


def draw_graph(graph: nx.Graph, mask_nodes_image: np.ndarray, graph_image: np.ndarray) -> None:
    draw_edges(
        graph=graph,
        graph_image=graph_image,
    )

    draw_nodes(
        graph=graph,
        mask_nodes_image=mask_nodes_image,
        graph_image=graph_image,
    )


def main() -> None:
    od.download("https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset")
    original_mask = cv2.imread("massachusetts-roads-dataset/tiff/train_labels/10378735_15.tif")
    mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2GRAY)

    _, mask_subplot = plt.subplots()
    _, mask_nodes_subplot = plt.subplots()
    _, graph_subplot = plt.subplots()

    harris_result = cv2.cornerHarris(
        src=mask,
        blockSize=3,  # Size of neighborhood (empirical)
        ksize=9,  # Sobel parameter (empirical)
        k=0.18  # Empirical value
    )

    graph = nx.Graph()

    add_nodes(
        harris_result=harris_result,
        threshold=100,  # Empirical value
        graph=graph
    )

    add_edges(
        mask=mask,
        threshold=60,
        graph=graph
    )

    height, width = harris_result.shape
    mask_nodes_image: np.ndarray = original_mask.copy()
    graph_image: np.ndarray = np.full(shape=(height, width, 3), dtype=np.uint8, fill_value=(180, 180, 180))

    draw_graph(graph=graph, mask_nodes_image=mask_nodes_image, graph_image=graph_image)

    mask_subplot.imshow(original_mask)
    mask_nodes_subplot.imshow(mask_nodes_image)
    graph_subplot.imshow(graph_image)

    cv2.imwrite("mask_nodes.png", cv2.cvtColor(mask_nodes_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite("graph.png", cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR))

    plt.show()


if __name__ == '__main__':
    main()
