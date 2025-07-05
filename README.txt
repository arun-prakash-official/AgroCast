AgroCast Project Folder

1. Place your dataset of plant images inside the `dataset/` folder.
   - Example structure:
     dataset/
       Tomato___Healthy/
         image1.jpg
       Tomato___Early_blight/
         image2.jpg

2. Run `train_model.py` to generate `plant_disease_model.h5`
3. Then run `app.py` using Streamlit:

   streamlit run app.py

Enjoy detecting plant diseases with voice guidance! 🌿
