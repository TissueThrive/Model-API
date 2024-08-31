from fastapi import APIRouter

from services.plantStatusIdentification import predictPlantHealth
from services.callusIdentification import predictCallus
from services.callusShape import predictCallusShape
from services.callusColour import predictCallustColor
from services.callusArea import predictCallusArea

router = APIRouter()

router.post("/plant_status_identification")(predictPlantHealth)

router.post("/callus_identification")(predictCallus)
router.post("/callus_shape_identification")(predictCallusShape)
router.post("/callus_color_identification")(predictCallustColor)
router.post("/callus_area_calculation")(predictCallusArea)