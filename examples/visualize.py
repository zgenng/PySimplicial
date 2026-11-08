from pysimplicial import gscat
from pysimplicial.datasets import projective_plane
from pysimplicial.visualization import draw_complex_and_cover, draw_surface_3d


def main():
    complex_ = projective_plane()
    result = gscat(complex_)
    draw_complex_and_cover(complex_, result, complex_name="RP²", legend=True)
    draw_surface_3d(complex_, result.cover, complex_name="RP²", gscat_value=result.value)


if __name__ == "__main__":
    main()
