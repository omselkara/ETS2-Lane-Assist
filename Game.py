from ets2sdktelemetry import *

ets = Ets2SdkTelemetry()

def get_game_brake():
    return ets.ets2telemetry.Controls.GameBrake  #0

def get_game_steer():
    return -ets.ets2telemetry.Controls.GameSteer #1

def get_game_throttle():
    return ets.ets2telemetry.Controls.GameThrottle #2

def get_user_brake():
    return ets.ets2telemetry.Controls.UserBrake #3

def get_user_steer():
    return -ets.ets2telemetry.Controls.UserSteer #4

def get_user_throttle():
    return ets.ets2telemetry.Controls.UserThrottle #5

def get_acc_x():
    return ets.ets2telemetry.Physics.AccelerationX #6

def get_acc_y():
    return ets.ets2telemetry.Physics.AccelerationY #7

def get_acc_z():
    return ets.ets2telemetry.Physics.AccelerationZ #8

def get_rot_x():
    return ets.ets2telemetry.Physics.RotationX #9

def get_rot_y():
    return ets.ets2telemetry.Physics.RotationY #10

def get_rot_z():
    return ets.ets2telemetry.Physics.RotationZ #11

def get_speed():
    return ets.ets2telemetry.Physics.Speed #12

def get_speed_kmh():
    return ets.ets2telemetry.Physics.SpeedKmh #13

def get_speed_mph():
    return ets.ets2telemetry.Physics.SpeedMph #14

def is_cc_active():
    if ets.ets2telemetry.DriveTrain.CruiseControl: #15
        return 1
    return 0

def get_all_data():
    return [get_game_brake(),get_game_steer(),get_game_throttle(),get_user_brake(),get_user_steer(),get_user_throttle(),get_acc_x(),get_acc_y(),get_acc_z(),get_rot_x(),get_rot_y(),get_rot_z(),get_speed(),get_speed_kmh(),get_speed_mph(),is_cc_active()]


screen_pos = (740,440,1410,630)#670,190

