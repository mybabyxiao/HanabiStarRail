from webbrowser import get
import cv2
import numpy as np
import pygetwindow
import psutil
import win32api
import win32con
import win32gui
import time
import sys
from PyQt5.QtWidgets import QApplication
import pyautogui
import pygetwindow as gw

# 截图


def window_screenshot(window):
    # 窗口边界
    left = window.left
    top = window.top
    width = window.width
    height = window.height

    # 截图
    bring_window_to_foreground(win32gui.FindWindow(None, hwnd.title))
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    # screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    # 创建目录
    # os.makedirs('./temp/screenshot/', exist_ok=True)
    # 保存截图
    # 保存
    # cv2.imwrite('./temp/screenshot/screenshot.png', screenshot)
    screenshot.save('./temp/screenshot/screenshot.png')
    # 使用 cv2 读取截图并获取分辨率
    img = cv2.imread('./temp/screenshot/screenshot.png')
    resolution = (img.shape[1], img.shape[0])  # (width, height)
    print(f"截图分辨率: {resolution}")


# 查找窗口


def find_window(title):

    # 获取所有窗口
    windows = gw.getAllTitles()
    print("当前打开的窗口列表：")
    for window in windows:
        print(window)
    target_window = gw.getWindowsWithTitle(title)

    if target_window:
        print(f"找到窗口：{title}")
        return target_window[0]  # 获取第一个匹配的窗口信息
    else:
        print("未找到该窗口！")

# 将窗口置于前台


def bring_window_to_foreground(title):
    """
    将指定窗口置于前台
    :param hwnd: 窗口句柄
    """
    # 判断窗口是否最小化，如果是则恢复窗口
    if win32gui.IsIconic(win32gui.FindWindow(None, hwnd.title)):
        win32gui.ShowWindow(win32gui.FindWindow(
            None, hwnd.title), win32con.SW_RESTORE)
    # 设置窗口到前台
    win32gui.SetForegroundWindow(win32gui.FindWindow(None, hwnd.title))

# 发送鼠标点击


def send_mouse_click(hwnd, x, y):
    """
    模拟鼠标点击到指定窗口的坐标
    :param hwnd: 窗口句柄
    :param x: 相对窗口的 x 坐标
    :param y: 相对窗口的 y 坐标
    """
    # 转换坐标到屏幕坐标
    rect = win32gui.GetWindowRect(hwnd)
    screen_x = rect[0] + x
    screen_y = rect[1] + y

    # 发送鼠标点击消息
    win32api.SetCursorPos((screen_x, screen_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,
                         screen_x, screen_y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, screen_x, screen_y, 0, 0)

# 发送键盘输入


def send_keyboard_input(hwnd, text):
    """
    向指定窗口发送键盘输入
    :param hwnd: 窗口句柄
    :param text: 要输入的文本
    """
    pass

# 三通道彩色匹配


def Three_channel_matching(templatepath, targetpath):
    # 加载彩色模板图片和目标图片
    template = cv2.imread(templatepath)  # 模板图片（彩色）
    target = cv2.imread(targetpath)      # 目标图片（彩色）

    # 检查图片是否加载成功
    if template is None or target is None:
        raise Exception("图片加载失败，请检查文件路径！")

    # 转换为三个通道（BGR）的图像
    # 提取每个通道的特征
    template_b, template_g, template_r = cv2.split(template)
    target_b, target_g, target_r = cv2.split(target)

    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()

    # 分别对每个通道提取特征
    kp_b, des_b = sift.detectAndCompute(template_b, None)
    kp_g, des_g = sift.detectAndCompute(template_g, None)
    kp_r, des_r = sift.detectAndCompute(template_r, None)

    # 对目标图像的每个通道提取特征
    kp_b_target, des_b_target = sift.detectAndCompute(target_b, None)
    kp_g_target, des_g_target = sift.detectAndCompute(target_g, None)
    kp_r_target, des_r_target = sift.detectAndCompute(target_r, None)

    # 设置 FLANN 参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 检索次数，值越大精度越高但速度越慢

    # 创建 FLANN 匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用 KNN 进行匹配
    matches_b = flann.knnMatch(des_b, des_b_target, k=2)
    matches_g = flann.knnMatch(des_g, des_g_target, k=2)
    matches_r = flann.knnMatch(des_r, des_r_target, k=2)

    # 过滤匹配点
    good_matches = []
    for m, n in matches_b:
        if m.distance < 0.75 * n.distance:  # 距离比
            good_matches.append(m)
    for m, n in matches_g:
        if m.distance < 0.75 * n.distance:  # 距离比
            good_matches.append(m)
    for m, n in matches_r:
        if m.distance < 0.75 * n.distance:  # 距离比
            good_matches.append(m)

    # 至少需要4个点来计算单应性
    if len(good_matches) > 4:
        # 提取匹配点的位置
        src_pts = np.float32(
            [kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_b_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 使用单应性矩阵将模板的边界映射到目标图片
        h, w, _ = template.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # # 绘制匹配框在目标图片上
        # cv2.polylines(target, [np.int32(dst)], True,
        #               (0, 255, 0), 3, cv2.LINE_AA)
        # 各个点的坐标
        center_loc = tuple([np.int32((np.int32(dst[0][0][0])+np.int32(dst[2][0][0]))/2),
                            np.int32((np.int32(dst[0][0][1])+np.int32(dst[2][0][1]))/2)])
        top_left_loc = tuple(np.int32(dst[0][0]))
        top_right_loc = tuple(np.int32(dst[1][0]))
        bottom_right_loc = tuple(np.int32(dst[2][0]))
        bottom_left_loc = tuple(np.int32(dst[3][0]))
        # # 输出左上角坐标
        # print(f"匹配到的左上角坐标:{tuple(np.int32(dst[0][0]))}")
        # print(f"匹配到的右上角坐标:{tuple(np.int32(dst[1][0]))}")
        # print(f"匹配到的右下角坐标:{tuple(np.int32(dst[2][0]))}")
        # print(f"匹配到的左下角坐标:{tuple(np.int32(dst[3][0]))}")
        # print(f"匹配到的中心点坐标:{center_loc}")

        # # 显示结果
        # cv2.imshow("Matching Result", target)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return top_left_loc, top_right_loc, bottom_right_loc, bottom_left_loc, center_loc
    else:
        return None

# 灰度匹配


def Grayscale_matching(templatepath, targetpath, matches_threshold=50, distance_threshold=0.5):
    # 加载模板图片和目标图片（转为灰度图）
    template = cv2.imread(templatepath, cv2.IMREAD_GRAYSCALE)  # 模板图片（灰度）
    target = cv2.imread(targetpath, cv2.IMREAD_GRAYSCALE)      # 目标图片（灰度）

    # 检查图片是否加载成功
    if template is None or target is None:
        raise Exception("图片加载失败，请检查文件路径！")

    # 创建 SIFT 检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    # 设置 FLANN 参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 检索次数，值越大精度越高但速度越慢

    # 创建 FLANN 匹配器
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用 KNN 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 过滤匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < distance_threshold * n.distance:  # 距离比
            good_matches.append(m)

    # 至少需要4个点来计算单应性
    if len(good_matches) > matches_threshold:
        # 提取匹配点的位置
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 使用单应性矩阵将模板的边界映射到目标图片
        h, w = template.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # # 绘制匹配框在目标图片上
        # cv2.polylines(target, [np.int32(dst)], True,
        #               (0, 255, 0), 3, cv2.LINE_AA)
        # 各个点的坐标
        center_loc = tuple([np.int32((np.int32(dst[0][0][0])+np.int32(dst[2][0][0]))/2),
                            np.int32((np.int32(dst[0][0][1])+np.int32(dst[2][0][1]))/2)])
        top_left_loc = tuple(np.int32(dst[0][0]))
        top_right_loc = tuple(np.int32(dst[1][0]))
        bottom_right_loc = tuple(np.int32(dst[2][0]))
        bottom_left_loc = tuple(np.int32(dst[3][0]))
        # # 输出左上角坐标
        # print(f"匹配到的左上角坐标:{tuple(np.int32(dst[0][0]))}")
        # print(f"匹配到的右上角坐标:{tuple(np.int32(dst[1][0]))}")
        # print(f"匹配到的右下角坐标:{tuple(np.int32(dst[2][0]))}")
        # print(f"匹配到的左下角坐标:{tuple(np.int32(dst[3][0]))}")
        # print(f"匹配到的中心点坐标:{center_loc}")

        # # 显示结果
        # cv2.imshow("Matching Result", target)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return top_left_loc, top_right_loc, bottom_right_loc, bottom_left_loc, center_loc
    else:
        return None


def get_status(hwnd):

    bring_window_to_foreground(hwnd)
    #
    time.sleep(1)
    window_screenshot(hwnd)
    time.sleep(1)

    a = Grayscale_matching('./resource/template/loading.png',
                           './temp/screenshot/screenshot.png')
    b = Grayscale_matching('./resource/template/home_page.png',
                           './temp/screenshot/screenshot.png')
    print(a, b)


def get_start_button():
    hwnd = find_window("崩坏：星穹铁道")
    if not hwnd:
        print("未找到窗口！")
        sys.exit()
    return hwnd


hwnd = get_start_button()
get_status(hwnd)
