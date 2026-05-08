#!/usr/bin/env bash
# Camera YOLO Logger — 一键安装脚本
#   克隆仓库后执行:  bash install.sh
#   安装后全局可用:  camera-yolo [flags]

set -euo pipefail

# ── 检测项目根目录（本脚本所在目录的上级，或当前目录）──────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR=""

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -f "$PWD/pyproject.toml" ]; then
    PROJECT_DIR="$PWD"
fi

if [ -z "$PROJECT_DIR" ]; then
    echo "错误: 找不到 pyproject.toml，请在项目根目录运行此脚本"
    exit 1
fi

BIN_DIR="$HOME/.local/bin"
WRAPPER="$BIN_DIR/camera-yolo"

# ── 创建 ~/.local/bin ────────────────────────────────────────
mkdir -p "$BIN_DIR"

# ── 创建 wrapper 脚本 ────────────────────────────────────────
cat > "$WRAPPER" << WRAPPER_EOF
#!/usr/bin/env bash
# Camera YOLO Logger — 自动定位项目并调用 uv run
#   由 install.sh 生成，项目路径: $PROJECT_DIR

PROJECT_DIR="$PROJECT_DIR"

if [ ! -d "\$PROJECT_DIR" ]; then
    echo "错误: 项目目录不存在: \$PROJECT_DIR" >&2
    echo "请重新克隆 termux-camera-yolo 并运行 install.sh" >&2
    exit 1
fi

if [ ! -f "\$PROJECT_DIR/pyproject.toml" ]; then
    echo "错误: 项目目录不完整，缺少 pyproject.toml" >&2
    exit 1
fi

cd "\$PROJECT_DIR"
exec uv run camera-yolo "\$@"
WRAPPER_EOF

chmod +x "$WRAPPER"
echo "已创建: $WRAPPER"

# ── 确保 ~/.local/bin 在 PATH 中 ────────────────────────────
ensure_path() {
    local rc_file="$1"
    if [ -f "$rc_file" ]; then
        if ! grep -q '.local/bin' "$rc_file" 2>/dev/null; then
            echo >> "$rc_file"
            echo '# Added by camera-yolo-logger install.sh' >> "$rc_file"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc_file"
            echo "已添加 PATH 到: $rc_file"
        fi
    fi
}

ensure_path "$HOME/.bashrc"
ensure_path "$HOME/.zshrc"
ensure_path "$HOME/.profile"

# ── 使当前会话生效 ──────────────────────────────────────────
export PATH="$BIN_DIR:$PATH"

# ── 验证安装 ─────────────────────────────────────────────────
if command -v camera-yolo >/dev/null 2>&1; then
    echo ""
    echo "安装完成! camera-yolo 已全局可用"
    echo ""
    echo "用法:"
    echo "  camera-yolo                    # 一次性检测 (text 输出)"
    echo "  camera-yolo --json             # JSON 输出"
    echo "  camera-yolo --json --classes person car"
    echo "  camera-yolo --monitor --json --motion"
    echo "  camera-yolo --server --server-port 5000"
    echo ""
    echo "如果命令未找到，请执行:  source ~/.bashrc  或  source ~/.zshrc"
else
    echo ""
    echo "安装完成，但需要重新加载 shell 配置:"
    echo "  source ~/.bashrc    # bash 用户"
    echo "  source ~/.zshrc     # zsh 用户"
    echo "然后就可以直接使用:  camera-yolo"
fi
