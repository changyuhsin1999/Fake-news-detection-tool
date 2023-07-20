from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class OriginalTextForm(FlaskForm):
	original_text = TextAreaField('Original Text', validators=[Length(min=20, max=10000)], render_kw={'placeholder': 'Paste your text here or press generate..'})
	generate = SubmitField('Generate Text')
	predict = SubmitField('Predict')
