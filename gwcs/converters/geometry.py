"""
ASDF tags for geometry related models.

"""

from asdf_astropy.converters.transform.core import TransformConverterBase

__all__ = ["DirectionCosinesConverter", "SphericalCartesianConverter"]


class DirectionCosinesConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/direction_cosines-*",)
    types = (
        "gwcs.geometry.ToDirectionCosines",
        "gwcs.geometry.FromDirectionCosines",
    )

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.geometry import FromDirectionCosines, ToDirectionCosines

        match transform_type := node.get("transform_type"):
            case "to_direction_cosines":
                return ToDirectionCosines()

            case "from_direction_cosines":
                return FromDirectionCosines()

        msg = f"Unknown transform_type {transform_type}"
        raise TypeError(msg)

    def to_yaml_tree_transform(self, model, tag, ctx):
        from gwcs.geometry import FromDirectionCosines, ToDirectionCosines

        match model:
            case FromDirectionCosines():
                transform_type = "from_direction_cosines"

            case ToDirectionCosines():
                transform_type = "to_direction_cosines"

            case _:
                msg = f"Model of type {model.__class__} is not supported."
                raise TypeError(msg)

        return {"transform_type": transform_type}


class SphericalCartesianConverter(TransformConverterBase):
    tags = ("tag:stsci.edu:gwcs/spherical_cartesian-*",)
    types = (
        "gwcs.geometry.SphericalToCartesian",
        "gwcs.geometry.CartesianToSpherical",
    )

    def from_yaml_tree_transform(self, node, tag, ctx):
        from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

        wrap_lon_at = node["wrap_lon_at"]

        match transform_type := node.get("transform_type"):
            case "spherical_to_cartesian":
                return SphericalToCartesian(wrap_lon_at=wrap_lon_at)

            case "cartesian_to_spherical":
                return CartesianToSpherical(wrap_lon_at=wrap_lon_at)

        msg = f"Unknown transform_type {transform_type}"
        raise TypeError(msg)

    def to_yaml_tree_transform(self, model, tag, ctx):
        from gwcs.geometry import CartesianToSpherical, SphericalToCartesian

        match model:
            case SphericalToCartesian():
                transform_type = "spherical_to_cartesian"

            case CartesianToSpherical():
                transform_type = "cartesian_to_spherical"

            case _:
                msg = f"Model of type {model.__class__} is not supported."
                raise TypeError(msg)

        return {"transform_type": transform_type, "wrap_lon_at": model.wrap_lon_at}
