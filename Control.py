import ctypes,time,keyboard,win32gui,win32api
import pyautogui as pag
from winsound import Beep
from pynput.keyboard import Key, Controller
MOUSEEVENTF_MOVE = 0x0001 #Movement occurred.
MOUSEEVENTF_LEFTDOWN = 0x0002 #The left button was pressed.
MOUSEEVENTF_LEFTUP = 0x0004 #The left button was released.
MOUSEEVENTF_RIGHTDOWN = 0x0008 #The right button was pressed.
MOUSEEVENTF_RIGHTUP = 0x0010 #The right button was released.
MOUSEEVENTF_MIDDLEDOWN = 0x0020	#The middle button was pressed.
MOUSEEVENTF_MIDDLEUP = 0x0040 #The middle button was released.
MOUSEEVENTF_XDOWN = 0x0080 #An X button was pressed.
MOUSEEVENTF_XUP = 0x0100 #An X button was released.
MOUSEEVENTF_WHEEL = 0x0800 #The wheel was moved, if the mouse has a wheel. The amount of movement is specified in mouseData.
MOUSEEVENTF_HWHEEL = 0x1000 #The wheel was moved horizontally, if the mouse has a wheel. The amount of movement is specified in mouseData.
MOUSEEVENTF_MOVE_NOCOALESCE = 0x2000 #The WM_MOUSEMOVE messages will not be coalesced. The default behavior is to coalesce WM_MOUSEMOVE messages.
MOUSEEVENTF_VIRTUALDESK = 0x4000 #Maps coordinates to the entire desktop. Must be used with MOUSEEVENTF_ABSOLUTE.
MOUSEEVENTF_ABSOLUTE = 0x8000 #The dx and dy members contain normalized absolute coordinates
WHEEL_DELTA = 120 #One wheel click one scroll
KEYEVENTF_EXTENDEDKEY = 0x0001 #If specified, the scan code was preceded by a prefix byte that has the value 0xE0 (224).
KEYEVENTF_KEYUP = 0x0002 #If specified, the key is being released. If not specified, the key is being pressed.
KEYEVENTF_SCANCODE = 0x0008 #If specified, wScan identifies the key and wVk is ignored.
KEYEVENTF_UNICODE = 0x0004 #If specified, the system synthesizes a VK_PACKET keystroke. The wVk parameter must be zero
NULL = ctypes.c_ulong(0)
PUL = ctypes.POINTER(ctypes.c_ulong)
KEY_CODES = "https://wiki.nexusmods.com/index.php/DirectX_Scancodes_And_How_To_Use_Them"

def GetRValue(col):
    return col&255
def GetGValue(col):
    return (col>>8)&255
def GetBValue(col):
    return col>>16
def GetRGB(col):
    """
    Returns (R,G,B)
    """
    return (GetRValue(col),GetGValue(col),GetBValue(col))
def GetRGBValue(r,g,b):
    """
    Returns Combined RGB Value
    """
    return (b<<16)|(g<<8)|( r )
class KeyBoardInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwNULLInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwNULLInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBoardInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

KEYS = {
"1":2,
"2":3,
"3":4,
"4":5,
"5":6,
"6":7,
"7":8,
"8":9,
"9":10,
"0":11,
"Q":16,
"W":17,
"E":18,
"R":19,
"T":20,
"Y":21,
"U":22,
"I":23,
"O":24,
"P":25,
"A":30,
"S":31,
"D":32,
"F":33,
"G":34,
"H":35,
"J":36,
"K":37,
"L":38,
"Z":44,
"X":45,
"C":46,
"V":47,
"B":48,
"N":49,
"M":50,
" ":57,
"UP":200,
"LEFT":203,
"RIGHT":205,
"DOWN":208,
"ENTER":28
}

def check_fail(pos):
    if sum(pos)==0:
        raise Exception("Fail Safe")
class PC:
    def __init__(self,fail_safe=True):
        self.keyboard = Controller()        
        self.user32 = ctypes.windll.user32
        self.res = (self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1))
        self.rs = (65536/self.res[0],65536/self.res[1]);
        self.fail_safe = fail_safe
    #MOUSE
    def get_pos(self):
        '''
        Get Cursor Position    
        '''
        return win32gui.GetCursorPos()
    def set_pos(self, x, y):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        '''
        Set Cursor Position    
        '''
        x = int(x * self.rs[0])
        y = int(y * self.rs[1])
        ii_ = Input_I()
        ii_.mi = MouseInput(x, y, 0, (MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE), 0, ctypes.pointer(NULL))
        command = Input(ctypes.c_ulong(0), ii_)
        self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
        
    def add_pos(self, x, y):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        '''
        Move Cursor Position Relative
        '''
        x = int(x)
        y = int(y)
        ii_ = Input_I()
        ii_.mi = MouseInput(x, y, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(NULL))
        command = Input(ctypes.c_ulong(0), ii_)
        self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    def mouse_event(self, typ="left"):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        '''
        Types are:
            left        : Press and Release Left Button
            right       : Press and Release Right Button
            middle      : Press and Release Middle Button
            left_down   : Press Left Button
            right_down  : Press Right Button
            middle_down : Press Middle Button
            left_up     : Relase Left Button
            right_up    : Relase Right Button
            middle_up   : Relase Middle 
        '''
        event = 0
        if typ=="left":
            event = MOUSEEVENTF_LEFTDOWN|MOUSEEVENTF_LEFTUP
        elif typ=="right":
            event = MOUSEEVENTF_RIGHTDOWN|MOUSEEVENTF_RIGHTUP
        elif typ=="middle":
            event = MOUSEEVENTF_MIDDLEDOWN|MOUSEEVENTF_MIDDLEUP
        elif typ=="left_down":
            event = MOUSEEVENTF_LEFTDOWN
        elif typ=="left_up":
            event = MOUSEEVENTF_LEFTUP
        elif typ=="right_down":
            event = MOUSEEVENTF_RIGHTDOWN
        elif typ=="right_up":
            event = MOUSEEVENTF_RIGHTUP
        elif typ=="middle_down":
            event = MOUSEEVENTF_MIDDLEDOWN
        elif typ=="middle_up":
            event = MOUSEEVENTF_MIDDLEUP
        else:
            raise Exception(f"Button \"{typ}\" Not Found")
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, event, 0, ctypes.pointer(NULL))
        command = Input(ctypes.c_ulong(0), ii_)
        self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    def click(self, x=-1, y=-1, button="left"):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        '''
        Types are:
            left        : Press and Release Left Button
            right       : Press and Release Right Button
            middle      : Press and Release Middle Button
        '''
        if type(x)==str:
            if x=="left" or x=="right" or x=="middle":
                self.mouse_event(x)
                return
            else:
                raise Exception(f"Button \"{button}\" Not Found")
        elif x!=-1 and y!=-1:
            self.set_pos(x,y)
        self.mouse_event(button)
        
    def scroll(self,value):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        if value>1 or value <-1:
            raise Exception(f"Value Must Be In Range(-1,+1) Got {value}")
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, int(value*WHEEL_DELTA), MOUSEEVENTF_WHEEL, 0, ctypes.pointer(NULL))
        command = Input(ctypes.c_ulong(0), ii_)
        self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
        
    #MONITOR
    def get_pixel(self,x=-1,y=-1):
        """
        Get Pixel At X,Y Default Is Mouse Positon
        """
        if x==-1 or y==-1:
            x,y = self.get_pos()
        dc = win32gui.GetDC(0)
        value = GetRGB(win32gui.GetPixel(dc,x,y))
        win32gui.ReleaseDC(0,dc)
        return value
    
    def set_pixel(self,x,y,r,g=-1,b=-1):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        color = r
        if g!=-1 and b==-1:
            color = GetRGBValue(r,g,b)
        dc = win32gui.GetDC(0)
        win32gui.SetPixel(dc,x,y,color)
        win32gui.ReleaseDC(0,dc)

    #KEYBOARD
    def key_down(self,key):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        self.keyboard.press(key)
##        typ = type(key)
##        if typ==str:
##            try:
##                val = KEYS[key.upper()]
##            except KeyError:
##                raise Exception(f"Key \"{key}\" Not Found")
##            ii_ = Input_I()
##            ii_.ki = KeyBoardInput(0,val,KEYEVENTF_SCANCODE,0,ctypes.pointer(NULL))
##            command = Input(ctypes.c_ulong(1), ii_)
##            self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
##        elif typ==int:
##            ii_ = Input_I()
##            ii_.ki = KeyBoardInput(0,key,KEYEVENTF_SCANCODE,0,ctypes.pointer(NULL))
##            command = Input(ctypes.c_ulong(1), ii_)
##            self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
##        else:
##            raise Exception(f"Key \"{key}\" Not Found")
    def key_up(self,key):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        self.keyboard.release(key)        
##        typ = type(key)
##        if typ==str:
##            try:
##                val = KEYS[key.upper()]
##            except KeyError:
##                raise Exception(f"Key \"{key}\" Not Found")
##            ii_ = Input_I()
##            ii_.ki = KeyBoardInput(0,val,KEYEVENTF_SCANCODE|KEYEVENTF_KEYUP,0,ctypes.pointer(NULL))
##            command = Input(ctypes.c_ulong(1), ii_)
##            self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
##        elif typ==int:
##            ii_ = Input_I()
##            ii_.ki = KeyBoardInput(0,key,KEYEVENTF_SCANCODE|KEYEVENTF_KEYUP,0,ctypes.pointer(NULL))
##            command = Input(ctypes.c_ulong(1), ii_)
##            self.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
##        else:
##            raise Exception(f"Key \"{key}\" Not Found")
    def press(self,key):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        self.keyboard.press(key)
        self.keyboard.release(key)

    def write(self,text):
        pos = self.get_pos()
        if self.fail_safe:
            check_fail(pos)
        for i in text:
            self.keyboard.press(i)
            self.keyboard.release(i)
            time.sleep(0.001)
        
    def wait(self,key,beep=0):
        if beep!=0:
            Beep(1000,beep)
        while not self.is_down(key):
            time.sleep(0.1)
        if beep!=0:
            Beep(1000,beep)
    def is_down(self,key):
        return keyboard.is_pressed(key)
