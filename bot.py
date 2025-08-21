import random
import psutil
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Токен вашего бота, полученный от @BotFather
TOKEN = "5461281270:AAEBJ_EeONrpc-WyHb_KkFBXnPlmkCalsPQ"


# Функция для команды /start
async def start(update, context):
    await update.message.reply_text(
        "Привет! Я бот, который отвечает случайным числом до 10,000,000. Напиши что угодно!")


# Функция для обработки текстовых сообщений
async def handle_message(update, context):
    # random_number = random.randint(0, 10000000)
    await update.message.reply_text(str(f"{psutil.cpu_percent(interval=1)}"))


def main():
    # Создаём приложение
    application = Application.builder().token(TOKEN).build()

    # Добавляем обработчик команды /start
    application.add_handler(CommandHandler("start", start))

    # Добавляем обработчик текстовых сообщений (игнорируем команды)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем бота
    application.run_polling()


if __name__ == "__main__":
    main()
