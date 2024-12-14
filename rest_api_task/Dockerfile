FROM python:3.10.12
EXPOSE 5000
WORKDIR /app

# Install Flask
RUN pip install flask

# Copy the application code
COPY . /app
# RUN cp -r ~/.vscode-server/extensions ./extensions
# # Copy all VS Code extensions
# COPY ./extensions /root/.vscode-server/extensions
# RUN rm -rf extensions

CMD ["flask", "run", "--host", "0.0.0.0"]