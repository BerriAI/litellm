"use client";

import React, { useState } from "react";
import { Grid, Col, Icon } from "@tremor/react";
import { Title } from "@tremor/react";
import {
    Modal,
    message,
} from "antd";
import { modelDeleteCall } from "./networking";
import { TrashIcon } from "@heroicons/react/outline";

interface DeleteModelProps {
    modelID: string;
    accessToken: string;
    callback?: ()=>void;
    setModelData: (data: any) => void;
    modelData: any;
}

const DeleteModelButton: React.FC<DeleteModelProps> = ({
    modelID,
    accessToken,
    callback,
    setModelData,
    modelData
}) => {
     const [isModalVisible, setIsModalVisible] = useState(false);
     const [modelToDelete, setModelToDelete] = useState<string | null>(null);

    const handleDelete = async () => {
        if (modelToDelete == null || modelData == null || accessToken == null) {
            return;
        }

        try {
            message.info("Making API Call");
            const response = await modelDeleteCall(accessToken, modelToDelete);
            console.log("model delete Response:", response);
            
            const filteredData = modelData.data.filter(
                (item: any) => item.model_info.id !== modelID
            );
            
            setModelData({
                ...modelData,
                data: filteredData
            });

            message.success(`Model ${modelToDelete} deleted successfully`);
            
        } catch (error) {
            console.error("Error deleting the model:", error);
            message.error("Failed to delete model");
        } finally {
            setIsModalVisible(false);
            setModelToDelete(null);
        }
    };

    return (
        <div>
            <Icon
                onClick={() => {
                    setIsModalVisible(true);
                    setModelToDelete(modelID);
                }}
                icon={TrashIcon}
                size="sm"
            />

            <Modal
                open={isModalVisible}
                onOk={handleDelete}
                okType="danger"
                onCancel={() => setIsModalVisible(false)}
            >
                <Grid numItems={1} className="gap-2 w-full">

                    <Title>Delete Model</Title>
                    <Col numColSpan={1}>
                        <p>
                            Are you sure you want to delete this model? This action is irreversible.
                        </p>
                    </Col>
                    <Col numColSpan={1}>
                        <p>
                            Model ID: <b>{modelID}</b>
                        </p>
                    </Col>
                </Grid>
            </Modal>
        </div>
    );
};

export default DeleteModelButton;