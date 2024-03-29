import * as React from "react";

export as namespace CopyToClipboard;

declare class CopyToClipboard extends React.PureComponent<CopyToClipboard.Props> {}

declare namespace CopyToClipboard {
    class CopyToClipboard extends React.PureComponent<Props> {}

    interface Options {
        debug?: boolean | undefined;
        message?: string | undefined;
        format?: string | undefined; // MIME type
    }

    interface Props {
        children?: React.ReactNode;
        text: string;
        onCopy?(text: string, result: boolean): void;
        options?: Options | undefined;
    }
}

export = CopyToClipboard;
