#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import os
import sys
import time
import asyncio
import subprocess
import threading
import json
import tempfile
import logging
import websockets
import ssl as ssl_module
import wave
import signal
import atexit
from pathlib import Path
from typing import Optional
from queue import Queue

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FunASR API Service")

# 全局变量
backend_ready = False
backend_process = None


def start_backend_service():
    """启动后端 ASR 服务"""
    global backend_ready, backend_process
    
    cmd = (
        'cd /workspace/FunASR/runtime && '
        'bash run_server.sh '
        '--download-model-dir /workspace/models '
        '--vad-dir iic/speech_fsmn_vad_zh-cn-16k-common-onnx '
        '--model-dir iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch '
        '--punc-dir iic/punc_ct-transformer_cn-en-common-vocab471067-large-onnx '
        '--lm-dir iic/speech_ngram_lm_zh-cn-ai-wesp-fst '
        '--itn-dir thuduj12/fst_itn_zh '
        '--hotword /workspace/models/hotwords.txt'
    )
    
    logger.info("正在启动后端 ASR 服务...")
    
    # 使用 preexec_fn 创建新的进程组，这样可以终止整个进程树
    backend_process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        preexec_fn=os.setsid  # 创建新的进程组
    )
    
    # 监控日志输出
    for line in backend_process.stdout:
        logger.info(f"后端服务: {line.strip()}")
        if "asr model init finished. listen on port" in line:
            logger.info("后端 ASR 服务启动成功！")
            backend_ready = True
            break
    
    # 继续输出剩余日志（在后台线程中）
    def log_output():
        for line in backend_process.stdout:
            logger.info(f"后端服务: {line.strip()}")
    
    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()


async def call_asr_service(
    audio_path: str,
    host: str = "127.0.0.1",
    port: int = 10095,
    mode: str = "offline",
    use_ssl: bool = True,
    use_itn: bool = True,
    hotword: str = "",
    chunk_size: list = None,
    chunk_interval: int = 10,
    audio_fs: int = 16000
) -> dict:
    """
    调用 ASR websocket 服务
    
    Args:
        audio_path: 音频文件路径
        host: websocket 服务地址
        port: websocket 服务端口
        mode: 识别模式 (offline, online, 2pass)
        use_ssl: 是否使用 SSL
        use_itn: 是否使用反向文本正则化
        hotword: 热词
        chunk_size: 分块大小
        chunk_interval: 分块间隔
        audio_fs: 音频采样率
    
    Returns:
        识别结果字典
    """
    # 记录开始时间
    start_time = time.time()
    
    if chunk_size is None:
        chunk_size = [5, 10, 5]
    
    # 读取音频文件
    wav_format = "pcm"
    if audio_path.endswith(".pcm"):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    elif audio_path.endswith(".wav"):
        with wave.open(audio_path, "rb") as wav_file:
            params = wav_file.getparams()
            audio_fs = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            audio_bytes = bytes(frames)
    else:
        wav_format = "others"
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    
    # 处理热词
    hotword_msg = ""
    if hotword.strip() != "":
        if os.path.exists(hotword):
            fst_dict = {}
            with open(hotword, encoding="utf-8") as f:
                hot_lines = f.readlines()
                for line in hot_lines:
                    words = line.strip().split(" ")
                    if len(words) >= 2:
                        try:
                            fst_dict[" ".join(words[:-1])] = int(words[-1])
                        except ValueError:
                            pass
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = hotword
    
    # 建立 websocket 连接
    if use_ssl:
        ssl_context = ssl_module.SSLContext()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl_module.CERT_NONE
        uri = f"wss://{host}:{port}"
    else:
        uri = f"ws://{host}:{port}"
        ssl_context = None
    
    logger.info(f"正在连接到 {uri}")
    
    results = []
    
    async with websockets.connect(
        uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
    ) as websocket:
        
        # 发送配置信息
        message = json.dumps({
            "mode": mode,
            "chunk_size": chunk_size,
            "chunk_interval": chunk_interval,
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 0,
            "audio_fs": audio_fs,
            "wav_name": os.path.basename(audio_path),
            "wav_format": wav_format,
            "is_speaking": True,
            "hotwords": hotword_msg,
            "itn": use_itn,
        })
        await websocket.send(message)
        
        # 分块发送音频数据
        stride = int(60 * chunk_size[1] / chunk_interval / 1000 * audio_fs * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        
        # 创建接收消息的任务
        async def receive_messages():
            while True:
                try:
                    msg = await websocket.recv()
                    msg_dict = json.loads(msg)
                    results.append(msg_dict)
                    logger.info(f"收到识别结果: {msg_dict.get('text', '')}")
                    
                    if msg_dict.get("is_final", False) and mode == "offline":
                        break
                except Exception as e:
                    logger.error(f"接收消息错误: {e}")
                    break
        
        recv_task = asyncio.create_task(receive_messages())
        
        # 发送音频数据
        for i in range(chunk_num):
            beg = i * stride
            data = audio_bytes[beg: beg + stride]
            await websocket.send(data)
            
            if i == chunk_num - 1:
                is_speaking = False
                message = json.dumps({"is_speaking": is_speaking})
                await websocket.send(message)
            
            sleep_duration = 0.001 if mode == "offline" else 60 * chunk_size[1] / chunk_interval / 1000
            await asyncio.sleep(sleep_duration)
        
        # 等待接收完所有消息
        if mode == "offline":
            await asyncio.sleep(1)
            await recv_task
        else:
            await asyncio.sleep(2)
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    logger.info(f"目标服务执行耗时: {elapsed_time:.3f} 秒")
    
    return {
        "success": True,
        "results": results,
        "wav_name": os.path.basename(audio_path),
        "service_elapsed_time": round(elapsed_time, 3)
    }


@app.post("/asr")
async def asr_recognize(
    mode: str = Form("offline"),
    use_itn: bool = Form(True),
    hotword: str = Form(""),
):
    """
    ASR 识别接口
    
    Args:
        mode: 识别模式 (offline, online, 2pass)
        use_itn: 是否使用反向文本正则化
        hotword: 热词字符串或热词文件路径
    
    Returns:
        识别结果
    """
    # 记录请求开始时间
    request_start_time = time.time()
    
    if not backend_ready:
        raise HTTPException(status_code=503, detail="后端 ASR 服务尚未就绪，请稍后再试")
    
    # 固定音频文件路径
    audio_path = "/workspace/test.mp3"
    
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {audio_path}")
    
    try:
        logger.info(f"开始识别音频文件: {audio_path}")
        
        # 调用 ASR 服务
        result = await call_asr_service(
            audio_path=audio_path,
            host="127.0.0.1",
            port=10095,
            mode=mode,
            use_ssl=True,
            use_itn=use_itn,
            hotword=hotword
        )
        
        # 计算总耗时
        total_elapsed_time = time.time() - request_start_time
        logger.info(f"API 服务完成请求总耗时: {total_elapsed_time:.3f} 秒")
        
        # 添加总耗时到返回结果
        result["total_elapsed_time"] = round(total_elapsed_time, 3)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"ASR 识别错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR 识别失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if backend_ready else "initializing",
        "backend_ready": backend_ready
    }


@app.on_event("startup")
async def startup_event():
    """应用启动时的事件处理"""
    logger.info("正在启动 FunASR API 服务...")
    
    # 在后台线程中启动后端服务
    backend_thread = threading.Thread(target=start_backend_service, daemon=True)
    backend_thread.start()
    
    # 等待后端服务就绪
    max_wait_time = 300  # 最多等待5分钟
    start_time = time.time()
    while not backend_ready and (time.time() - start_time) < max_wait_time:
        await asyncio.sleep(1)
    
    if not backend_ready:
        logger.error("后端服务启动超时！")
    else:
        logger.info("FunASR API 服务启动完成！")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的事件处理"""
    global backend_process
    if backend_process:
        logger.info("正在关闭后端服务...")
        try:
            # 终止整个进程组
            pgid = os.getpgid(backend_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            
            # 等待进程结束
            try:
                backend_process.wait(timeout=5)
                logger.info("后端服务已正常关闭")
            except subprocess.TimeoutExpired:
                logger.warning("后端服务未在5秒内关闭，强制终止")
                os.killpg(pgid, signal.SIGKILL)
                backend_process.wait()
                logger.info("后端服务已强制终止")
        except Exception as e:
            logger.error(f"关闭后端服务时出错: {e}")


def cleanup_backend_process():
    """清理后端进程（用于 atexit）"""
    global backend_process
    if backend_process and backend_process.poll() is None:
        logger.info("清理后端进程...")
        try:
            pgid = os.getpgid(backend_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            backend_process.wait(timeout=3)
        except Exception as e:
            logger.error(f"清理后端进程时出错: {e}")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except:
                pass


def signal_handler(signum, frame):
    """信号处理函数"""
    logger.info(f"收到信号 {signum}，正在退出...")
    cleanup_backend_process()
    sys.exit(0)


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

# 注册退出时的清理函数
atexit.register(cleanup_backend_process)


if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8190,
        log_level="info"
    )
