import streamlit as st
import face_recognition
import cv2
import numpy as np
import tempfile

"""
Сервис видеоидентификации клиентов.
Сравнивает лицо на загруженном фото с лицами, которые встречаются на видео.
"""

def get_face_encoding_from_image(image_file):
    """
    Принимает файл изображения (в формате bytes), 
    загружает его как объект numpy (BGR/RGB) 
    и возвращает face-encodings (векторные представления лиц).
    """
    # Преобразуем загруженный файл в массив numpy
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    
    # Декодируем как изображение в формате OpenCV (BGR)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Конвертируем BGR -> RGB для face_recognition
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Находим координаты лиц
    face_locations = face_recognition.face_locations(image_rgb)
    
    # Находим face-encodings
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
    
    return face_encodings

def process_video_and_compare(video_file, known_face_encodings, tolerance=0.6):
    """
    Принимает видео (в формате bytes) и список известных face-encodings.
    Извлекает несколько кадров из видео и сравнивает с этими encodings.
    Возвращает True, если в видео найден хотя бы один кадр с соответствующим лицом.
    """
    # Создаём временный файл для видео
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(video_file.read())
        temp_path = temp.name

    # Открываем видео через OpenCV
    cap = cv2.VideoCapture(temp_path)

    frame_interval = 10  # берем каждый 10-й кадр
    frame_count = 0
    found_match = False

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_face_locations = face_recognition.face_locations(frame_rgb)
            frame_face_encodings = face_recognition.face_encodings(frame_rgb, frame_face_locations)
            
            # Сравниваем каждое лицо на кадре с эталонными encodings
            for face_encoding in frame_face_encodings:
                results = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                if any(results):
                    found_match = True
                    break
            
            if found_match:
                break
        
        frame_count += 1

    cap.release()
    return found_match

def main():
    st.title("Сервис видеоидентификации клиентов")

    # Зарезервируем место для возможных сообщений (совпадение/не совпадение).
    match_message = st.empty()

    # Две колонки: левая (видео), правая (фото)
    col_left, col_right = st.columns(2)

    # -----------------------------
    # 1) Секция: Загрузка видео (слева) и фото (справа)
    # -----------------------------
    with col_left:
        st.subheader("Загрузите видео клиента")
        video_file = st.file_uploader("Видео (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
        # Если пользователь загрузил видео, выводим его сразу
        if video_file is not None:
            st.video(video_file, format="video/mp4")

    with col_right:
        st.subheader("Загрузите фото клиента")
        image_file = st.file_uploader("Фото (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        # Если пользователь загрузил фото, выводим его сразу
        if image_file is not None:
            st.image(image_file, caption="Фото клиента", use_container_width=True)

    # Разделительная линия
    st.write("---")
    
    # -----------------------------
    # 2) Кнопка проверки
    # -----------------------------
    if st.button("Проверить соответствие"):
        # Сбросим прошлое сообщение
        match_message.empty()

        if not video_file or not image_file:
            match_message.warning("Пожалуйста, загрузите и видео, и фото!")
        else:
            # Получаем face-encodings из фото
            face_encodings_from_image = get_face_encoding_from_image(image_file)

            if len(face_encodings_from_image) == 0:
                match_message.error("На загруженном фото не обнаружено лиц!")
            else:
                # Сравниваем с кадрами видео
                match_found = process_video_and_compare(video_file, face_encodings_from_image)

                if match_found:
                    match_message.success("Совпадение найдено: человек на видео соответствует фото клиента.")
                else:
                    match_message.error("Совпадений не обнаружено.")

if __name__ == "__main__":
    main()
