from ursina import *
from direct.stdpy import thread

from car import Car
from remote_controller import RemoteController
from raycast_sensor import *

from sun import SunLight

from track import Track

# Window
window.vsync = True # Set to false to uncap FPS limit of 60
app = Ursina(size=(1280,1024))
window.title = "Rally"
window.borderless = False
window.show_ursina_splash = False
window.cog_button.disable()
window.fps_counter.enable()
window.exit_button.disable()

#   Global models & textures
#                   car model       particle model    raycast model
global_models = [ "sports-car.obj", "particles.obj",  "line.obj"]
#                Car texture             Particle Textures
global_texs = [ "sports-red.png", "particle_forest_track.png", "red.png"]

# Starting new thread for assets
track_metadata = "assets/SlightlyHarder/track_metadata.json"
track = Track(track_metadata)
track.load_assets(global_models, global_texs)

# Car
car = Car()
car.sports_car()
# Tracks
car.set_track(track)

remote_controller = RemoteController(car = car)

car.multiray_sensor = MultiRaySensor(car, 15, 90)
car.multiray_sensor.enable()

# Lighting + shadows
sun = SunLight(direction = (-0.7, -0.9, 0.5), resolution = 3072, car = car)
ambient = AmbientLight(color = Vec4(0.5, 0.55, 0.66, 0) * 0.75)

render.setShaderAuto()

# Sky
Sky(texture = "sky")

car.visible = True
mouse.locked = True

car.enable()

car.camera_angle = "top"
car.change_camera = True
car.camera_follow = True

track.activate()
track.played = True


app.run()
